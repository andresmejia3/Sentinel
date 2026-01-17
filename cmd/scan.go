package cmd

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"runtime"
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
	scanCmd.Flags().IntVarP(&scanOpts.NumEngines, "engines", "e", runtime.NumCPU(), "Number of parallel engine workers")
	scanCmd.Flags().StringVarP(&scanOpts.GapDuration, "gap", "g", "2s", "Max absence duration before closing a track")
	scanCmd.Flags().Float64Var(&scanOpts.MatchThreshold, "threshold", 0.6, "Face matching distance threshold (lower is stricter)")

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

	// 3. Get FPS for Time Calculations
	fps, err := utils.GetVideoFPS(opts.InputPath)
	if err != nil {
		utils.Die("Failed to determine video FPS", err, nil)
	}

	// 4. Parse Gap Duration
	gap, err := time.ParseDuration(opts.GapDuration)
	if err != nil {
		utils.Die("Invalid gap duration format (use '2s', '500ms')", err, nil)
	}

	// 0. Get total frames for progress bar
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

	// 3. Spawn the Engine Pool
	for i := 0; i < opts.NumEngines; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			startWorker(workerID, taskChan, resultsChan)
		}(i)
	}

	// 4. Start FFmpeg
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

	// 5. Frame Splitter & Nth-Frame Logic
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

	// 6. Cleanup & Completion Check
	if err := ffmpeg.Wait(); err != nil {
		if stderrBuf.Len() > 0 {
			fmt.Fprintf(os.Stderr, "\nFFmpeg Logs:\n%s\n", stderrBuf.String())
		}
		utils.Die("FFmpeg execution failed", err, nil)
	}

	close(taskChan)
	wg.Wait()
	close(resultsChan)

	// Process and merge all results
	processResults(resultsChan, DB, videoID, fps, gap, opts)

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
func startWorker(id int, tasks <-chan types.FrameTask, results chan<- scanResult) {
	worker, err := worker.NewPythonWorker(id)
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
}

func processResults(results <-chan scanResult, db *store.Store, videoID string, fps float64, gap time.Duration, opts Options) {
	// Buffer for re-ordering frames (Worker 2 might finish before Worker 1)
	buffer := make(map[int]scanResult)
	nextFrame := opts.NthFrame // Assuming first frame is nthFrame based on loop logic

	// Tracking state
	var tracks []*activeTrack
	nextTrackID := 1
	maxGapFrames := int(gap.Seconds() * fps)

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
				bestMatch := -1
				minDist := opts.MatchThreshold

				for i, t := range tracks {
					dist := euclideanDist(face.Vec, t.MeanVec)
					if dist < minDist {
						minDist = dist
						bestMatch = i
					}
				}

				if bestMatch != -1 {
					// Update existing track
					t := tracks[bestMatch]
					t.LastFrame = frame.Index
					t.Count++
					for j := 0; j < 128; j++ {
						t.SumVec[j] += face.Vec[j]
						t.MeanVec[j] = t.SumVec[j] / float64(t.Count)
					}
				} else {
					// Create new track
					newT := &activeTrack{
						ID:         nextTrackID,
						StartFrame: frame.Index,
						LastFrame:  frame.Index,
						SumVec:     make([]float64, 128),
						MeanVec:    make([]float64, 128),
						Count:      1,
					}
					copy(newT.SumVec, face.Vec)
					copy(newT.MeanVec, face.Vec)
					tracks = append(tracks, newT)
					nextTrackID++
				}
			}

			// 2. Close stale tracks
			active := tracks[:0]
			for _, t := range tracks {
				if frame.Index-t.LastFrame > maxGapFrames {
					// Track ended, persist it
					startSec := float64(t.StartFrame) / fps
					endSec := float64(t.LastFrame) / fps
					db.InsertInterval(context.Background(), videoID, startSec, endSec, t.Count, t.MeanVec)
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
		db.InsertInterval(context.Background(), videoID, startSec, endSec, t.Count, t.MeanVec)
	}
}

func euclideanDist(a, b []float64) float64 {
	var sum float64
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return math.Sqrt(sum)
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
}
