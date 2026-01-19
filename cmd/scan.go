package cmd

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"

	"github.com/andresmejia3/sentinel/internal/store"
	"github.com/andresmejia3/sentinel/internal/types"
	"github.com/andresmejia3/sentinel/internal/utils"
	"github.com/andresmejia3/sentinel/internal/worker"
	"github.com/schollz/progressbar/v3"
	"github.com/spf13/cobra"
)

const megabyte = 1024 * 1024
const embeddingDim = 512

var scanOpts Options

var scanCmd = &cobra.Command{
	Use:   "scan",
	Short: "Scan video with parallel engines",
	Run: func(cmd *cobra.Command, args []string) {
		runScan(scanOpts)
	},
}

func init() {
	scanCmd.Flags().StringVarP(&scanOpts.InputPath, "input", "i", "", "Path to video")
	scanCmd.Flags().IntVarP(&scanOpts.NthFrame, "nth-frame", "n", 10, "AI keyframe interval (e.g. scan every 10th frame)")
	scanCmd.Flags().IntVarP(&scanOpts.NumEngines, "engines", "e", 1, "Number of parallel engine workers")
	scanCmd.Flags().StringVarP(&scanOpts.GracePeriod, "grace-period", "g", "2s", "The longest period where a face can be missing before Sentinel declares they are out of frame and logs it to the database")
	scanCmd.Flags().Float64VarP(&scanOpts.MatchThreshold, "threshold", "t", 0.6, "Face matching threshold (lower is stricter)")
	scanCmd.Flags().StringVarP(&scanOpts.BlipDuration, "blip-duration", "b", "100ms", "Minimum duration of a track to be considered valid (filters blips)")
	scanCmd.Flags().BoolVarP(&scanOpts.DebugScreenshots, "debug-screenshots", "d", false, "Save debug images with bounding boxes to /data/debug_frames/")

	scanCmd.MarkFlagRequired("input")
	rootCmd.AddCommand(scanCmd)

}

// Buffer pool to reduce GC pressure during scanning
var frameBufferPool = sync.Pool{
	New: func() interface{} { return make([]byte, 0, megabyte) },
}

// runScan orchestrates the video scanning process: DB setup, Worker Pool, FFmpeg streaming, and Progress tracking.
func runScan(opts Options) {
	validateScanFlags(&opts)

	// 1. Database is initialized in Root PersistentPreRun
	ctx := context.Background()

	// 2. Generate Video ID & Register
	videoID, err := utils.GenerateVideoID(opts.InputPath)
	if err != nil {
		utils.Die("Failed to generate video ID", err, nil)
	}
	if err := DB.EnsureVideoMetadata(ctx, videoID, opts.InputPath); err != nil {
		utils.Die("Failed to register video metadata", err, nil)
	}
	fmt.Fprintf(os.Stderr, "📼 Processing Video ID: %s\n", videoID[:12])
	fmt.Fprintf(os.Stderr, "⚙️  Spawning %d Worker Engines...\n", opts.NumEngines)

	// 3. Get FPS for Time Calculations
	fps, err := utils.GetVideoFPS(opts.InputPath)
	if err != nil {
		utils.Die("Failed to determine video FPS", err, nil)
	}

	// 5. Get total frames for progress bar
	totalVideoFrames := utils.GetTotalFrames(opts.InputPath)

	if totalVideoFrames <= 0 {
		// Fallback to a spinner or unknown total if ffprobe fails
		totalVideoFrames = -1
	}

	bar := progressbar.NewOptions(totalVideoFrames,
		progressbar.OptionSetDescription("🔍 Sentinel Scanning"),
		progressbar.OptionSetWriter(os.Stderr), // Write bar to Stderr
		progressbar.OptionShowCount(),
	)

	taskChan := make(chan types.FrameTask, opts.NumEngines)
	resultsChan := make(chan scanResult, opts.NumEngines*2)
	var wg sync.WaitGroup

	// 6. Start Aggregator (Consumer)
	// Must run concurrently to prevent deadlock on resultsChan
	aggDone := make(chan struct{})
	go func() {
		processResults(resultsChan, DB, videoID, fps, opts)
		close(aggDone)
	}()

	// 7. Spawn the Engine Pool
	for i := 0; i < opts.NumEngines; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			startWorker(workerID, taskChan, resultsChan, opts.DebugScreenshots)
		}(i)
	}

	// 8. Start FFmpeg
	ffmpeg := utils.NewFFmpegCmd(opts.InputPath)

	var stderrBuf bytes.Buffer
	ffmpeg.Stderr = &stderrBuf

	ffmpegOut, err := ffmpeg.StdoutPipe()
	if err != nil {
		utils.Die("Failed to create FFmpeg stdout pipe", err, nil)
	}
	defer ffmpegOut.Close() // Ensure pipe is closed to prevent leaks/zombies

	if err := ffmpeg.Start(); err != nil {
		utils.Die("Failed to start FFmpeg", err, nil)
	}

	// 9. Frame Splitter & Nth-Frame Logic
	scanner := bufio.NewScanner(ffmpegOut)
	scanner.Buffer(make([]byte, megabyte), 64*megabyte)
	scanner.Split(utils.SplitJpeg)

	totalFrames := 0
	sentFrames := 0
	for scanner.Scan() {
		totalFrames++
		bar.Add(1) // Update progress bar for every frame read

		if totalFrames%opts.NthFrame == 0 {
			// Get buffer from pool
			buf := frameBufferPool.Get().([]byte)
			if cap(buf) < len(scanner.Bytes()) {
				buf = make([]byte, len(scanner.Bytes()))
			}
			buf = buf[:len(scanner.Bytes())]
			copy(buf, scanner.Bytes())
			taskChan <- types.FrameTask{Index: totalFrames, Data: buf}
			sentFrames++
		}
	}

	// Check for scanner errors (e.g. token too long, unexpected EOF)
	if err := scanner.Err(); err != nil {
		utils.Die("Frame scanner failed", err, nil)
	}

	// 10. Cleanup & Completion Check
	if err := ffmpeg.Wait(); err != nil {
		if stderrBuf.Len() > 0 {
			fmt.Fprintf(os.Stderr, "\nFFmpeg Logs:\n%s\n", stderrBuf.String())
		}
		utils.Die("FFmpeg execution failed", err, nil)
	}

	close(taskChan)
	wg.Wait()
	close(resultsChan)

	// Wait for aggregator to finish processing
	<-aggDone

	bar.Finish()
	fmt.Fprintf(os.Stderr, "\n🏁 Scan Complete. Processed %d keyframes out of %d total.\n", sentFrames, totalFrames)
}

// scanResult wraps the output from a worker to be sent to the aggregator
type scanResult struct {
	Index int
	Faces []types.FaceResult
}

// startWorker manages the lifecycle of a single Python worker process.
// It reads tasks from the channel, sends them to Python, and persists the results to the DB.
func startWorker(id int, tasks <-chan types.FrameTask, results chan<- scanResult, debug bool) {
	worker, err := worker.NewPythonWorker(id, debug)
	if err != nil {
		utils.Die("Worker startup failed", err, nil)
	}
	defer worker.Close()

	for task := range tasks {
		resp, err := worker.ProcessFrame(task.Data)

		// Return buffer to pool immediately after sending
		frameBufferPool.Put(task.Data)

		if err != nil {
			// DRAIN: Wait for process to exit and capture final stderr logs
			worker.Close()
			utils.Die("Python crashed", err, worker.Cmd)
		}

		// Now we can actually USE the data!
		var faces []types.FaceResult
		if err := json.Unmarshal(resp, &faces); err != nil {
			// Check if it's a Python error object (e.g. {"error": "..."})
			var errorResult types.ErrorResult
			if json.Unmarshal(resp, &errorResult) == nil && errorResult.Error != "" {
				fmt.Fprintf(os.Stderr, "\n⚠️ Worker %d Logic Error: %s\n", id, errorResult.Error)
				// Send empty result to prevent aggregator deadlock
				results <- scanResult{Index: task.Index, Faces: nil}
				continue
			}

			// Genuine unmarshal failure (garbage data)
			fmt.Fprintf(os.Stderr, "\n⚠️ Worker %d JSON Malformed: %v\n", id, err)
			// Send empty result to prevent aggregator deadlock
			results <- scanResult{Index: task.Index, Faces: nil}
			continue
		}

		// Send to aggregator instead of DB
		results <- scanResult{Index: task.Index, Faces: faces}
	}
}

// --- Aggregation & Tracking Logic ---

type activeTrack struct {
	ID         int
	StartFrame int
	LastFrame  int
	SumVec     []float64
	MeanVec    []float64
	Count      int
	BestThumb  []byte  // Raw JPEG bytes of the best face
	MaxQuality float64 // Best quality score seen so far
}

type timeRange struct {
	Start float64
	End   float64
}

func processResults(results <-chan scanResult, db *store.Store, videoID string, fps float64, opts Options) {
	// Buffer for re-ordering frames (Worker 2 might finish before Worker 1)
	buffer := make(map[int]scanResult)
	nextFrame := opts.NthFrame // Assuming first frame is nthFrame based on loop logic

	// Tracking state
	var tracks []*activeTrack

	// Local cache for thumbnails found in THIS session (for the summary report)
	type thumbData struct {
		Data    []byte
		Quality float64
	}
	sessionThumbs := make(map[int]thumbData)

	summary := make(map[int][]timeRange)
	totalDetections := 0
	gracePeriod, _ := time.ParseDuration(opts.GracePeriod)
	blipDuration, _ := time.ParseDuration(opts.BlipDuration)
	maxGapFrames := int(gracePeriod.Seconds() * fps)
	if maxGapFrames < 1 {
		maxGapFrames = 1 // Ensure at least 1 frame gap to prevent instant closing
	}

	fmt.Fprintf(os.Stderr, "⚙️  Tracker initialized with Cosine Distance (Threshold: %.2f)\n", opts.MatchThreshold)

	for res := range results {
		buffer[res.Index] = res

		// Process frames in strict order
		for {
			frame, ok := buffer[nextFrame]
			if !ok {
				break
			}
			delete(buffer, nextFrame)

			// 1. Match faces to tracks
			for _, face := range frame.Faces {
				totalDetections++
				bestMatch := -1
				minDist := opts.MatchThreshold

				for i, t := range tracks {
					dist := cosineDist(face.Vec, t.MeanVec)
					if dist < minDist {
						minDist = dist
						bestMatch = i
					}
				}

				if bestMatch != -1 {
					// Update existing track if there's a match
					t := tracks[bestMatch]
					t.LastFrame = frame.Index
					t.Count++
					for j := 0; j < embeddingDim; j++ {
						t.SumVec[j] += face.Vec[j]
						t.MeanVec[j] = t.SumVec[j] / float64(t.Count)
					}
					// Update best thumbnail if this face is higher quality
					if face.Quality > t.MaxQuality {
						// Decode Base64 only when we find a better image
						if imgBytes, err := base64.StdEncoding.DecodeString(face.ThumbB64); err == nil && len(imgBytes) > 0 {
							t.MaxQuality = face.Quality
							t.BestThumb = imgBytes
						}
					}
				} else {
					// No active track matched. Try Re-ID against history.
					// Query DB for nearest neighbor using pgvector
					matchID, err := db.FindClosestIdentity(context.Background(), face.Vec, opts.MatchThreshold)
					if err != nil {
						utils.Die("DB Identity Lookup failed", err, nil)
					}

					if matchID != -1 {
						// Re-ID Successful: Resurrect existing Identity
						newT := newActiveTrack(matchID, frame.Index, face.Vec, face.ThumbB64, face.Quality)
						tracks = append(tracks, newT)
					} else {
						// Truly New Identity -> Create in DB immediately
						newID, err := db.CreateIdentity(context.Background(), face.Vec)
						if err != nil {
							utils.Die("Failed to create new identity", err, nil)
						}

						newT := newActiveTrack(newID, frame.Index, face.Vec, face.ThumbB64, face.Quality)
						tracks = append(tracks, newT)

					}
				}
			}

			// 2. Close stale tracks
			active := tracks[:0]
			for _, t := range tracks {
				if frame.Index-t.LastFrame > maxGapFrames {
					// Track ended, persist it
					startSec := float64(t.StartFrame) / fps
					endSec := float64(t.LastFrame) / fps

					// Filter short tracks (blips)
					if (endSec - startSec) < blipDuration.Seconds() {
						continue
					}

					if err := db.InsertInterval(context.Background(), videoID, startSec, endSec, t.Count, t.ID); err != nil {
						utils.Die(fmt.Sprintf("Failed to persist track %d", t.ID), err, nil)
					}
					summary[t.ID] = append(summary[t.ID], timeRange{Start: startSec, End: endSec})

					// Update Global Identity (Weighted Average)
					if err := db.UpdateIdentity(context.Background(), t.ID, t.MeanVec, t.Count); err != nil {
						utils.Die(fmt.Sprintf("Failed to update identity %d", t.ID), err, nil)
					}

					// Update session thumbnail if this track had a better one
					if cur, exists := sessionThumbs[t.ID]; !exists || t.MaxQuality > cur.Quality {
						sessionThumbs[t.ID] = thumbData{
							Data:    t.BestThumb,
							Quality: t.MaxQuality,
						}
					}
				} else {
					active = append(active, t)
				}
			}
			tracks = active
			nextFrame += opts.NthFrame
		}
	}

	// Flush remaining tracks
	for _, t := range tracks {
		startSec := float64(t.StartFrame) / fps
		endSec := float64(t.LastFrame) / fps

		// Filter short tracks (blips)
		if (endSec - startSec) < blipDuration.Seconds() {
			continue
		}

		if err := db.InsertInterval(context.Background(), videoID, startSec, endSec, t.Count, t.ID); err != nil {
			utils.Die(fmt.Sprintf("Failed to persist track %d", t.ID), err, nil)
		}
		summary[t.ID] = append(summary[t.ID], timeRange{Start: startSec, End: endSec})

		if err := db.UpdateIdentity(context.Background(), t.ID, t.MeanVec, t.Count); err != nil {
			utils.Die(fmt.Sprintf("Failed to update identity %d", t.ID), err, nil)
		}

		if cur, exists := sessionThumbs[t.ID]; !exists || t.MaxQuality > cur.Quality {
			sessionThumbs[t.ID] = thumbData{
				Data:    t.BestThumb,
				Quality: t.MaxQuality,
			}
		}
	}

	fmt.Fprintf(os.Stderr, "\n---------------------------------------------------------\n")
	fmt.Fprintf(os.Stderr, "📊 SCAN SUMMARY\n")
	fmt.Fprintf(os.Stderr, "---------------------------------------------------------\n")

	var ids []int
	for id := range summary {
		ids = append(ids, id)
	}
	sort.Ints(ids)

	for _, id := range ids {
		thumbNote := ""
		// Check session thumbs
		if thumb, ok := sessionThumbs[id]; ok && len(thumb.Data) > 0 {
			filename := fmt.Sprintf("identity_%d.jpg", id)
			outDir := filepath.Join("/data", "thumbnails", videoID)
			_ = os.MkdirAll(outDir, 0755)
			_ = os.WriteFile(filepath.Join(outDir, filename), thumb.Data, 0644)
			thumbNote = fmt.Sprintf("(See thumbnails/%s/%s)", videoID, filename)
		}

		fmt.Fprintf(os.Stderr, "\n👤 Identity %d Found: %s\n", id, thumbNote)
		for _, r := range summary[id] {
			fmt.Fprintf(os.Stderr, "   %s -> %s\n", fmtTime(r.Start), fmtTime(r.End))
		}
	}

	fmt.Fprintf(os.Stderr, "\n---------------------------------------------------------\n")
	fmt.Fprintf(os.Stderr, "👁️  Total Face Detections:   %d\n", totalDetections)
	fmt.Fprintf(os.Stderr, "---------------------------------------------------------\n")
}

func newActiveTrack(id, frameIndex int, vec []float64, thumbB64 string, quality float64) *activeTrack {
	// Decode initial thumbnail
	imgBytes, _ := base64.StdEncoding.DecodeString(thumbB64) // Ignore error, bytes will be empty if invalid

	t := &activeTrack{
		ID:         id,
		StartFrame: frameIndex,
		LastFrame:  frameIndex,
		SumVec:     make([]float64, embeddingDim),
		MeanVec:    make([]float64, embeddingDim),
		Count:      1,
		BestThumb:  imgBytes,
		MaxQuality: quality,
	}
	copy(t.SumVec, vec)
	copy(t.MeanVec, vec)
	return t
}

func cosineDist(a, b []float64) float64 {
	var dot, sumA, sumB float64
	for i := range a {
		dot += a[i] * b[i]
		sumA += a[i] * a[i]
		sumB += b[i] * b[i]
	}
	// Return 1.0 (max distance) if a vector is zero to avoid division by zero
	if sumA == 0 || sumB == 0 {
		return 1.0
	}
	return 1.0 - (dot / (math.Sqrt(sumA) * math.Sqrt(sumB)))
}

// validateScanFlags ensures all CLI arguments are valid before starting heavy processes.
func validateScanFlags(opts *Options) {
	info, err := os.Stat(opts.InputPath)
	if err != nil {
		if os.IsNotExist(err) {
			utils.Die("Input file does not exist", err, nil)
		}
		utils.Die("Unable to access input file", err, nil)
	}
	if info.IsDir() {
		utils.Die("Input path is a directory, expected a video file", nil, nil)
	}
	if opts.NthFrame < 1 {
		utils.Die("Invalid nth-frame interval", fmt.Errorf("must be >= 1, got %d", opts.NthFrame), nil)
	}
	if opts.NumEngines < 1 {
		opts.NumEngines = 1
	}
	if opts.MatchThreshold <= 0 || opts.MatchThreshold > 1.0 {
		utils.Die("Invalid match threshold", fmt.Errorf("must be between 0.0 and 1.0, got %f", opts.MatchThreshold), nil)
	}
	if _, err := time.ParseDuration(opts.GracePeriod); err != nil {
		utils.Die("Invalid grace-period format (use '2s', '500ms')", err, nil)
	}
}

func fmtTime(seconds float64) string {
	duration := time.Duration(seconds * float64(time.Second))
	h := int(duration.Hours())
	m := int(duration.Minutes()) % 60
	s := int(duration.Seconds()) % 60
	return fmt.Sprintf("%02d:%02d:%02d", h, m, s)
}
