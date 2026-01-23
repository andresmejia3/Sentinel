package cmd

import (
	"bufio"
	"bytes"
	"context"
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
	// Use RunE so we can return errors to the root command for proper exit codes
	RunE: func(cmd *cobra.Command, args []string) error {
		cmd.SilenceUsage = true // Don't show help text on runtime errors
		return runScan(cmd.Context(), scanOpts)
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
	scanCmd.Flags().Float64VarP(&scanOpts.DetectionThreshold, "detection-threshold", "D", 0.5, "Face detection confidence threshold (0.0 - 1.0)")

	scanCmd.Flags().StringVar(&scanOpts.WorkerTimeout, "worker-timeout", "30s", "Timeout for a worker to process a single frame")

	scanCmd.MarkFlagRequired("input")
	rootCmd.AddCommand(scanCmd)

}

// Buffer pool to reduce GC pressure during scanning
var frameBufferPool = sync.Pool{
	New: func() interface{} { return make([]byte, 0, megabyte) },
}

// runScan orchestrates the video scanning process: DB setup, Worker Pool, FFmpeg streaming, and Progress tracking.
func runScan(ctx context.Context, opts Options) error {
	// Create a cancellable context so we can stop all workers if one fails
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	// Error channel to catch failures from background goroutines
	errChan := make(chan error, opts.NumEngines+2)

	if err := validateScanFlags(&opts); err != nil {
		return err
	}

	// Initialize channels early so we can start workers immediately
	taskChan := make(chan types.FrameTask, opts.NumEngines)
	resultsChan := make(chan scanResult, opts.NumEngines*2)
	var wg sync.WaitGroup

	estMemGB := 0.5 + (float64(opts.NumEngines) * 1.2)
	fmt.Fprintf(os.Stderr, "⚙️  Spawning %d Worker Engines (Est. Memory: ~%.1f GB)...\n", opts.NumEngines, estMemGB)

	// Start Workers EARLY (Parallelize with FFprobe/DB checks)
	readyChan := make(chan bool, opts.NumEngines) // Buffered: Workers drop message and keep going
	for i := 0; i < opts.NumEngines; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			startWorker(ctx, workerID, taskChan, resultsChan, readyChan, opts, errChan)
		}(i)
	}

	// 2. Generate Video ID & Register
	videoID, err := utils.GenerateVideoID(opts.InputPath)
	if err != nil {
		utils.ShowError("Failed to generate video ID", err, nil)
		return err
	}
	if err := DB.EnsureVideoMetadata(ctx, videoID, opts.InputPath); err != nil {
		utils.ShowError("Failed to register video metadata", err, nil)
		return err
	}
	fmt.Fprintf(os.Stderr, "📼 Processing Video ID: %s\n", videoID[:12])

	// 3. Get FPS for Time Calculations
	fps, err := utils.GetVideoFPS(ctx, opts.InputPath)
	if err != nil {
		utils.ShowError("Failed to determine video FPS", err, nil)
		return err
	}

	// 5. Get total frames for progress bar
	totalVideoFrames := utils.GetTotalFrames(ctx, opts.InputPath)

	if totalVideoFrames <= 0 {
		// Fallback to a spinner or unknown total if ffprobe fails
		totalVideoFrames = -1
	}

	// Progress bar for worker startup
	workerBar := progressbar.NewOptions(opts.NumEngines,
		progressbar.OptionSetDescription("🚀 Warming Up AI Engines"),
		progressbar.OptionSetWriter(os.Stderr),
		progressbar.OptionShowCount(),
		progressbar.OptionClearOnFinish(),
	)

	// WAIT HERE: Ensure all workers are ready before starting the heavy scan
	for i := 0; i < opts.NumEngines; i++ {
		select {
		case <-readyChan:
			workerBar.Add(1)
		case err := <-errChan:
			return err // Exit immediately if a worker fails to start
		case <-ctx.Done():
			return ctx.Err()
		}
	}

	bar := progressbar.NewOptions(totalVideoFrames,
		progressbar.OptionSetDescription("🔍 Sentinel Scanning"),
		progressbar.OptionSetWriter(os.Stderr), // Write bar to Stderr
		progressbar.OptionShowCount(),
	)

	// 6. Start Aggregator (Consumer)
	// Must run concurrently to prevent deadlock on resultsChan
	aggDone := make(chan struct{})
	go func() {
		processResults(ctx, resultsChan, DB, videoID, fps, opts, errChan)
		close(aggDone)
	}()

	// 8. Start FFmpeg
	ffmpeg := utils.NewFFmpegCmd(ctx, opts.InputPath)

	var stderrBuf bytes.Buffer
	ffmpeg.Stderr = &stderrBuf

	ffmpegOut, err := ffmpeg.StdoutPipe()
	if err != nil {
		utils.ShowError("Failed to create FFmpeg stdout pipe", err, nil)
		return err
	}
	defer ffmpegOut.Close() // Ensure pipe is closed to prevent leaks/zombies

	if err := ffmpeg.Start(); err != nil {
		utils.ShowError("Failed to start FFmpeg", err, nil)
		return err
	}

	// 9. Frame Splitter & Nth-Frame Logic
	scanner := bufio.NewScanner(ffmpegOut)
	scanner.Buffer(make([]byte, megabyte), 64*megabyte)
	scanner.Split(utils.SplitJpeg)

	totalFrames := 0
	sentFrames := 0
	for scanner.Scan() {
		// Non-blocking check for errors from workers
		select {
		case err := <-errChan:
			return err // Return immediately; defer cancel() will stop workers
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

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

			select {
			case taskChan <- types.FrameTask{Index: totalFrames, Data: buf}:
				sentFrames++
			case err := <-errChan:
				return err
			case <-ctx.Done():
				return ctx.Err()
			}
		}
	}

	// Check for scanner errors (e.g. token too long, unexpected EOF)
	if err := scanner.Err(); err != nil {
		utils.ShowError("Frame scanner failed", err, nil)
		return err
	}

	// 10. Cleanup & Completion Check
	if err := ffmpeg.Wait(); err != nil {
		if stderrBuf.Len() > 0 {
			fmt.Fprintf(os.Stderr, "\nFFmpeg Logs:\n%s\n", stderrBuf.String())
		}
		utils.ShowError("FFmpeg execution failed", err, nil)
		return err
	}

	close(taskChan)
	wg.Wait()
	close(resultsChan)

	// Wait for aggregator to finish processing
	<-aggDone

	// Final check for any errors that occurred during shutdown
	select {
	case err := <-errChan:
		return err
	default:
	}

	bar.Finish()
	fmt.Fprintf(os.Stderr, "\n🏁 Scan Complete. Processed %d keyframes out of %d total.\n", sentFrames, totalFrames)
	return nil
}

// scanResult wraps the output from a worker to be sent to the aggregator
type scanResult struct {
	Index int
	Faces []types.FaceResult
}

// startWorker manages the lifecycle of a single Python worker process.
// It reads tasks from the channel, sends them to Python, and persists the results to the DB.
func startWorker(ctx context.Context, id int, tasks <-chan types.FrameTask, results chan<- scanResult, ready chan<- bool, opts Options, errChan chan<- error) {
	cfg := worker.Config{
		Debug:              opts.DebugScreenshots,
		DetectionThreshold: opts.DetectionThreshold,
	}
	readTimeout, err := time.ParseDuration(opts.WorkerTimeout)
	if err != nil {
		// This should have been caught by validateScanFlags, but as a fallback:
		readTimeout = 30 * time.Second
	}
	cfg.ReadTimeout = readTimeout
	worker, err := worker.NewPythonWorker(ctx, id, cfg)
	if err != nil {
		utils.ShowError("Worker startup failed", err, nil)
		select {
		case errChan <- err:
		default:
		}
		return
	}
	defer worker.Close()

	// Signal to the main thread that this worker is ready
	ready <- true

	for {
		select {
		case <-ctx.Done():
			return
		case task, ok := <-tasks:
			if !ok {
				return
			}
			faces, err := worker.ProcessFrame(task.Data)

			// Return buffer to pool immediately after sending
			frameBufferPool.Put(task.Data)

			if err != nil {
				// DRAIN: Wait for process to exit and capture final stderr logs
				worker.Close()
				utils.ShowError("Python crashed", err, worker.Cmd)
				select {
				case errChan <- err:
				default:
				}
				return
			}

			// Send to aggregator instead of DB
			select {
			case results <- scanResult{Index: task.Index, Faces: faces}:
			case <-ctx.Done():
				return
			}
		}
	}
}

// --- Aggregation & Tracking Logic ---

type activeTrack struct {
	ID          int
	StartFrame  int
	LastFrame   int
	MeanVec     []float64
	Count       int
	BestThumb   []byte  // Raw JPEG bytes of the best face
	BestQuality float64 // Best quality score seen so far
	Name        string  // Display name (e.g. "Jenny" or "Identity 1")
	IsKnown     bool    // Is this an existing identity from the DB?
}

type timeRange struct {
	Start float64
	End   float64
}

func processResults(ctx context.Context, results <-chan scanResult, db *store.Store, videoID string, fps float64, opts Options, errChan chan<- error) {
	// Buffer for re-ordering frames (Worker 2 might finish before Worker 1)
	buffer := make(map[int]scanResult)
	nextFrame := opts.NthFrame // Assuming first frame is nthFrame based on loop logic

	// Tracking state
	var tracks []*activeTrack

	// Cache of the best quality score seen for each identity to avoid overwriting good thumbnails with bad ones.
	// Maps IdentityID -> MaxQuality
	bestQuality := make(map[int]float64)
	idNames := make(map[int]string)
	newlyCreated := make(map[int]bool) // Track which IDs were generated in this session
	tempIDCounter := -1                // Negative IDs for pending tracks

	summary := make(map[int][]timeRange)
	totalDetections := 0

	// WaitGroup to ensure all async persistence tasks finish before we exit
	var consumerWg sync.WaitGroup
	var producerWg sync.WaitGroup

	gracePeriod, _ := time.ParseDuration(opts.GracePeriod)
	blipDuration, _ := time.ParseDuration(opts.BlipDuration)
	maxGapFrames := int(gracePeriod.Seconds() * fps)
	if maxGapFrames < 1 {
		maxGapFrames = 1 // Ensure at least 1 frame gap to prevent instant closing
	}

	// Determine output directory base
	// If /data exists (Docker volume), use it. Otherwise use relative "data" (Local).
	outputBase := "data"
	if _, err := os.Stat("/data"); err == nil {
		outputBase = "/data"
	}
	// Optimization: Ensure output directory exists ONCE, not per-track
	unknownDir := filepath.Join(outputBase, "unknown", videoID)
	if err := os.MkdirAll(unknownDir, 0755); err != nil {
		utils.ShowError("Failed to create output directory", err, nil)
		select {
		case errChan <- err:
		default:
		}
		return
	}
	// Log the location so the user knows where to look
	fmt.Fprintf(os.Stderr, "📂 Output Directory: %s\n", unknownDir)

	// Async Thumbnail Writer (Sequential to prevent race conditions)
	// We use a buffered channel to offload disk I/O without blocking the main loop,
	// while ensuring that writes for the same ID happen in order.
	type thumbOp struct {
		id   int
		data []byte
	}
	thumbChan := make(chan thumbOp, 100)
	consumerWg.Add(1)
	go func() {
		defer consumerWg.Done()
		for op := range thumbChan {
			filename := fmt.Sprintf("identity_%d.jpg", op.id)
			path := filepath.Join(unknownDir, filename)

			// Retry logic: Attempt write 3 times to handle transient errors (e.g. file locks)
			var err error
			for i := 0; i < 3; i++ {
				if err = os.WriteFile(path, op.data, 0644); err == nil {
					break
				}
				time.Sleep(10 * time.Millisecond)
			}
			if err != nil {
				fmt.Fprintf(os.Stderr, "⚠️  Failed to write thumbnail for ID %d: %v\n", op.id, err)
			}
		}
	}()

	// Ensure cleanup happens even if we return early due to error (e.g. DB failure)
	defer func() {
		// 1. Wait for all DB updates to finish
		producerWg.Wait()
		// 2. Close the channel to signal the consumer
		close(thumbChan)
		// 3. Wait for the consumer to finish draining the channel
		consumerWg.Wait()
	}()

	fmt.Fprintf(os.Stderr, "⚙️  Tracker initialized with Cosine Distance (Threshold: %.2f)\n", opts.MatchThreshold)

	// Helper closure to persist a track (DRY: Used in loop and at flush)
	persistTrack := func(t *activeTrack) {
		startSec := float64(t.StartFrame) / fps
		endSec := float64(t.LastFrame) / fps

		// Filter short tracks (blips)
		if (endSec - startSec) < blipDuration.Seconds() {
			// Optimization: Since we use deferred creation (negative IDs),
			// we simply do nothing here. The track was never in the DB.
			return
		}

		// 1. Update In-Memory State (Synchronous)
		// Fix: Check quality synchronously to avoid map race conditions and ensure we only write better thumbs.
		shouldWriteThumb := false
		isNewIdentity := t.ID < 0
		finalID := t.ID

		if !isNewIdentity { // Known Identity
			curQ, exists := bestQuality[t.ID]
			if !exists || t.BestQuality > curQ {
				bestQuality[t.ID] = t.BestQuality
				shouldWriteThumb = true
			}
		} else {
			// CRITICAL FIX: Create Identity Synchronously.
			// If we do this async, a race condition exists where the person reappears
			// before the DB commit, causing a duplicate identity.
			var err error
			finalID, err = db.CreateIdentity(ctx, t.MeanVec, t.Count)
			if err != nil {
				utils.ShowError("Failed to create deferred identity", err, nil)
				select {
				case errChan <- err:
				default:
				}
				return
			}
			// Update metadata so the final summary report is correct
			idNames[finalID] = fmt.Sprintf("Identity %d", finalID)
			t.ID = finalID                       // Update track ID so summary grouping works
			bestQuality[finalID] = t.BestQuality // Fix: Record quality so we don't overwrite with worse thumbs later
			newlyCreated[finalID] = true         // Mark as new for the summary report
			shouldWriteThumb = true
		}

		// Fix: Send to channel SYNCHRONOUSLY to guarantee order.
		// The channel buffer (100) handles backpressure if disk is slow.
		if shouldWriteThumb {
			select {
			case thumbChan <- thumbOp{id: finalID, data: t.BestThumb}:
			case <-ctx.Done():
				return
			}
		}

		// 2. Offload I/O to Background (Async)
		producerWg.Add(1)
		go func(id int, vec []float64, count int, start, end float64, isNew bool) {
			defer producerWg.Done()

			if !isNew {
				// Only update vector if it was already known (accumulate average).
				// For new identities, CreateIdentity already inserted the vector.
				if err := db.UpdateIdentity(ctx, id, vec, count); err != nil {
					utils.ShowError(fmt.Sprintf("Failed to update identity %d", id), err, nil)
					select {
					case errChan <- err:
					default:
					}
					return
				}
			}

			if err := db.InsertInterval(ctx, videoID, start, end, count, id); err != nil {
				utils.ShowError(fmt.Sprintf("Failed to persist interval for %d", id), err, nil)
				select {
				case errChan <- err:
				default:
				}
				return
			}
		}(finalID, t.MeanVec, t.Count, startSec, endSec, isNewIdentity)

		// Update summary with the final ID.
		// Since we resolved the ID synchronously above, this is guaranteed to be the correct DB ID.
		summary[t.ID] = append(summary[t.ID], timeRange{Start: startSec, End: endSec})
	}

Loop:
	for {
		select {
		case <-ctx.Done():
			return // Exit immediately on cancel, triggering defers
		case res, ok := <-results:
			if !ok {
				break Loop
			}
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

						// Incrementally update the mean vector to save memory (avoids storing SumVec)
						k := float64(t.Count)
						for j := 0; j < embeddingDim; j++ {
							t.MeanVec[j] = (k*t.MeanVec[j] + face.Vec[j]) / (k + 1.0)
						}
						t.Count++

						// Update best thumbnail if this face is higher quality
						if face.Quality > t.BestQuality {
							t.BestQuality = face.Quality
							t.BestThumb = face.Thumb
						}
					} else {
						// No active track matched. Try Re-ID against history.
						// Query DB for nearest neighbor using pgvector
						matchID, matchName, err := db.FindClosestIdentity(ctx, face.Vec, opts.MatchThreshold)
						if err != nil {
							utils.ShowError("DB Identity Lookup failed", err, nil)
							select {
							case errChan <- err:
							default:
							}
							return
						}

						if matchID != -1 {
							// Re-ID Successful: Resurrect existing Identity
							newT := newActiveTrack(matchID, frame.Index, face.Vec, face.Thumb, face.Quality, matchName, true)
							tracks = append(tracks, newT)
							idNames[matchID] = matchName
						} else {
							// Truly New Identity -> Use Temporary Negative ID
							// We defer DB creation until the track survives the blip filter.
							tempID := tempIDCounter
							tempIDCounter--

							name := fmt.Sprintf("Identity %d (Pending)", tempID)
							newT := newActiveTrack(tempID, frame.Index, face.Vec, face.Thumb, face.Quality, name, false)
							tracks = append(tracks, newT)
							idNames[tempID] = name

						}
					}
				}

				// 2. Close stale tracks
				active := tracks[:0]
				for _, t := range tracks {
					if frame.Index-t.LastFrame > maxGapFrames {
						persistTrack(t)
					} else {
						active = append(active, t)
					}
				}
				tracks = active
				nextFrame += opts.NthFrame
			}
		}
	}

	// Flush remaining tracks
	for _, t := range tracks {
		persistTrack(t)
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
		// Check if we saved a thumbnail for this ID
		if _, ok := bestQuality[id]; ok {
			filename := fmt.Sprintf("identity_%d.jpg", id)
			thumbNote = fmt.Sprintf("(See unknown/%s/%s)", videoID, filename)
		}

		name := idNames[id]
		if name == "" {
			name = fmt.Sprintf("Identity %d", id)
		}

		status := "[EXISTING]"
		if newlyCreated[id] {
			status = "[NEW]"
		}
		fmt.Fprintf(os.Stderr, "\n👤 %s %s Found: %s\n", name, status, thumbNote)
		for _, r := range summary[id] {
			fmt.Fprintf(os.Stderr, "   %s -> %s\n", fmtTime(r.Start), fmtTime(r.End))
		}
	}

	fmt.Fprintf(os.Stderr, "\n---------------------------------------------------------\n")
	fmt.Fprintf(os.Stderr, "👁️  Total Face Detections:   %d\n", totalDetections)
	fmt.Fprintf(os.Stderr, "---------------------------------------------------------\n")
}

func newActiveTrack(id, frameIndex int, vec []float64, thumb []byte, quality float64, name string, isKnown bool) *activeTrack {
	t := &activeTrack{
		ID:          id,
		StartFrame:  frameIndex,
		LastFrame:   frameIndex,
		MeanVec:     make([]float64, embeddingDim),
		Count:       1,
		BestThumb:   thumb,
		BestQuality: quality,
		Name:        name,
		IsKnown:     isKnown,
	}
	copy(t.MeanVec, vec)
	return t
}

// cosineDist calculates the cosine distance between two vectors.
// It assumes vector 'a' is already normalized to unit length (a performance optimization).
func cosineDist(a, b []float64) float64 {
	// BCE (Bounds Check Elimination) Hint:
	// Proves to the compiler that a and b are large enough, removing checks inside the loop.
	if len(a) != len(b) || len(a) == 0 {
		return 1.0
	}
	_ = a[len(a)-1]
	_ = b[len(b)-1]

	var dot, sumB float64
	for i := range a {
		dot += a[i] * b[i]
		sumB += b[i] * b[i]
	}
	// 'a' is normalized from Python, so sumA is approx 1.0. We skip calculating it.
	if sumB == 0 {
		return 1.0
	}
	return 1.0 - (dot / math.Sqrt(sumB))
}

// validateScanFlags ensures all CLI arguments are valid before starting heavy processes.
func validateScanFlags(opts *Options) error {
	info, err := os.Stat(opts.InputPath)
	if err != nil {
		if os.IsNotExist(err) {
			utils.ShowError("Input file does not exist", err, nil)
			return err
		}
		utils.ShowError("Unable to access input file", err, nil)
		return err
	}
	if info.IsDir() {
		err := fmt.Errorf("is a directory")
		utils.ShowError("Input path is a directory, expected a video file", err, nil)
		return err
	}
	if opts.NthFrame < 1 {
		err := fmt.Errorf("must be >= 1, got %d", opts.NthFrame)
		utils.ShowError("Invalid nth-frame interval", err, nil)
		return err
	}
	if opts.NumEngines < 1 {
		opts.NumEngines = 1
	}
	if opts.MatchThreshold <= 0 || opts.MatchThreshold > 1.0 {
		err := fmt.Errorf("must be between 0.0 and 1.0, got %f", opts.MatchThreshold)
		utils.ShowError("Invalid match threshold", err, nil)
		return err
	}
	if _, err := time.ParseDuration(opts.GracePeriod); err != nil {
		utils.ShowError("Invalid grace-period format (use '2s', '500ms')", err, nil)
		return err
	}
	if _, err := time.ParseDuration(opts.WorkerTimeout); err != nil {
		utils.ShowError("Invalid worker-timeout format (use '30s', '1m')", err, nil)
		return err
	}
	return nil
}

func fmtTime(seconds float64) string {
	duration := time.Duration(seconds * float64(time.Second))
	h := int(duration.Hours())
	m := int(duration.Minutes()) % 60
	s := int(duration.Seconds()) % 60
	return fmt.Sprintf("%02d:%02d:%02d", h, m, s)
}
