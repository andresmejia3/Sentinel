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
	"sync/atomic"
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
var scanBufferSize int

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
	scanCmd.Flags().IntVarP(&scanBufferSize, "buffer-size", "B", 200, "Max number of frames to buffer in memory")

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

	// Flow Control: Prevent OOM by limiting in-flight frames.
	// Limits buffering in the aggregator if a worker stalls.
	if scanBufferSize < 1 {
		scanBufferSize = 1
	}
	inflightSem := make(chan struct{}, scanBufferSize)

	// Memory Monitoring
	var peakMemory uint64    // Atomic
	var currentMemory uint64 // Atomic
	pidChan := make(chan int, opts.NumEngines)

	go func() {
		var pids []int
		// Collect PIDs from workers as they start
		for i := 0; i < opts.NumEngines; i++ {
			select {
			case pid := <-pidChan:
				pids = append(pids, pid)
			case <-ctx.Done():
				return
			}
		}

		updateMem := func() {
			var total uint64
			total += utils.GetProcessRSS(os.Getpid()) // Go Process
			for _, pid := range pids {
				total += utils.GetProcessRSS(pid) // Python Workers
			}
			if currentPeak := atomic.LoadUint64(&peakMemory); total > currentPeak {
				atomic.StoreUint64(&peakMemory, total)
			}
			atomic.StoreUint64(&currentMemory, total)
		}

		updateMem()

		// Monitor Memory Loop
		ticker := time.NewTicker(250 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				updateMem()
			case <-ctx.Done():
				return
			}
		}
	}()

	// Start Workers EARLY (Parallelize with FFprobe/DB checks)
	readyChan := make(chan bool, opts.NumEngines)
	for i := 0; i < opts.NumEngines; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			startWorker(ctx, workerID, taskChan, resultsChan, readyChan, opts, errChan, pidChan)
		}(i)
	}

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

	fps, err := utils.GetVideoFPS(ctx, opts.InputPath)
	if err != nil {
		utils.ShowError("Failed to determine video FPS", err, nil)
		return err
	}

	totalVideoFrames := utils.GetTotalFrames(ctx, opts.InputPath)

	if totalVideoFrames <= 0 {
		totalVideoFrames = -1
	}

	workerBar := progressbar.NewOptions(opts.NumEngines,
		progressbar.OptionSetDescription("🚀 Warming Up AI Engines"),
		progressbar.OptionSetWriter(os.Stderr),
		progressbar.OptionShowCount(),
		progressbar.OptionClearOnFinish(),
	)

	// Wait for all workers to be ready before starting the heavy scan
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

	// Start Aggregator (Consumer) concurrently to prevent deadlock on resultsChan
	aggDone := make(chan struct{})
	finalIntervalsChan := make(chan []store.IntervalData, 1)
	go func() {
		processResults(ctx, resultsChan, DB, videoID, fps, opts, errChan, finalIntervalsChan, inflightSem)
		close(aggDone)
	}()

	go func() {
		updateDesc := func() {
			mem := atomic.LoadUint64(&currentMemory)
			bufLen := len(inflightSem)
			bufCap := cap(inflightSem)
			if mem > 0 {
				bar.Describe(fmt.Sprintf("🔍 Sentinel Scanning (RAM: %.2f GB | Buffer: %d/%d)", float64(mem)/(1024*1024*1024), bufLen, bufCap))
			}
		}
		updateDesc()

		ticker := time.NewTicker(250 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				updateDesc()
			}
		}
	}()

	ffmpeg := utils.NewFFmpegCmd(ctx, opts.InputPath, opts.NthFrame)

	var stderrBuf bytes.Buffer
	ffmpeg.Stderr = &stderrBuf

	ffmpegOut, err := ffmpeg.StdoutPipe()
	if err != nil {
		utils.ShowError("Failed to create FFmpeg stdout pipe", err, nil)
		return err
	}
	defer ffmpegOut.Close()

	if err := ffmpeg.Start(); err != nil {
		utils.ShowError("Failed to start FFmpeg", err, nil)
		return err
	}

	// Ensure we reap the process even if we return early
	ffmpegWaited := false
	defer func() {
		if !ffmpegWaited {
			ffmpeg.Wait()
		}
	}()

	scanner := bufio.NewScanner(ffmpegOut)
	scanner.Buffer(make([]byte, megabyte), 64*megabyte)
	scanner.Split(utils.SplitJpeg)

	scannedFrames := 0
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

		scannedFrames++
		// Since FFmpeg is skipping frames, every frame we read is a "hit".
		// We calculate the virtual index based on the count.
		// e.g. 1st frame read -> Index 0 (if N=10, select=not(mod(n,10)))
		virtualIndex := (scannedFrames - 1) * opts.NthFrame

		bar.Add(opts.NthFrame) // Advance bar by N for every 1 frame read

		buf := frameBufferPool.Get().([]byte)
		if cap(buf) < len(scanner.Bytes()) {
			buf = make([]byte, len(scanner.Bytes()))
		}
		buf = buf[:len(scanner.Bytes())]
		copy(buf, scanner.Bytes())

		// Acquire semaphore (Block if too many frames are in flight)
		select {
		case inflightSem <- struct{}{}:
		case <-ctx.Done():
			return ctx.Err()
		}

		select {
		case taskChan <- types.FrameTask{Index: virtualIndex, Data: buf}:
			sentFrames++
		case err := <-errChan:
			return err
		case <-ctx.Done():
			return ctx.Err()
		}
	}

	if err := scanner.Err(); err != nil {
		utils.ShowError("Frame scanner failed", err, nil)
		return err
	}

	ffmpegWaited = true // Mark as waited so defer doesn't run
	if err := ffmpeg.Wait(); err != nil {
		if stderrBuf.Len() > 0 {
			fmt.Fprintf(os.Stderr, "\nFFmpeg Logs:\n%s\n", stderrBuf.String())
		}
		utils.ShowError("FFmpeg execution failed", err, nil)
		return err
	}

	bar.Finish()

	close(taskChan)

	// Instead of blocking on wg.Wait(), we wait in a goroutine and select on it.
	wgDone := make(chan struct{})
	go func() {
		wg.Wait()
		close(wgDone)
	}()

	select {
	case <-wgDone:
		// Workers finished normally
	case err := <-errChan:
		return err
	case <-ctx.Done():
		return ctx.Err()
	}

	close(resultsChan)

	<-aggDone

	intervals := <-finalIntervalsChan

	// Atomic Commit: All intervals are inserted in a single transaction.
	fmt.Fprintf(os.Stderr, "🗄️  Committing %d intervals to database...\n", len(intervals))
	if err := DB.CommitScan(ctx, videoID, intervals); err != nil {
		utils.ShowError("Failed to commit scan results", err, nil)
		return err
	}

	// Final check for any errors that occurred during shutdown
	select {
	case err := <-errChan:
		return err
	default:
	}

	fmt.Fprintf(os.Stderr, "\n🏁 Scan Complete. Processed %d keyframes.\n", sentFrames)

	finalPeak := atomic.LoadUint64(&peakMemory)
	if finalPeak > 0 {
		fmt.Fprintf(os.Stderr, "🧠 Peak Memory Used: %.2f GB\n", float64(finalPeak)/(1024*1024*1024))
	}
	return nil
}

// scanResult wraps the output from a worker to be sent to the aggregator
type scanResult struct {
	Index int
	Faces []types.FaceResult
}

// startWorker manages the lifecycle of a single Python worker process.
// It reads tasks from the channel, sends them to Python, and persists the results to the DB.
func startWorker(ctx context.Context, id int, tasks <-chan types.FrameTask, results chan<- scanResult, ready chan<- bool, opts Options, errChan chan<- error, pidChan chan<- int) {
	cfg := worker.ScanConfig{
		Debug:              opts.DebugScreenshots,
		DetectionThreshold: opts.DetectionThreshold,
	}
	readTimeout, err := time.ParseDuration(opts.WorkerTimeout)
	if err != nil {
		// This should have been caught by validateScanFlags, but as a fallback:
		readTimeout = 30 * time.Second
	}
	cfg.ReadTimeout = readTimeout
	worker, err := worker.NewPythonScanWorker(ctx, id, cfg)
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
	pidChan <- worker.Cmd.Process.Pid
	ready <- true

	for {
		select {
		case <-ctx.Done():
			return
		case task, ok := <-tasks:
			if !ok {
				return
			}
			faces, err := worker.ProcessScanFrame(task.Data)

			// Return buffer to pool immediately after sending
			frameBufferPool.Put(task.Data)

			if err != nil {
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
	ID         int
	MasterID   int // Optimization: Store MasterID to avoid DB lookups during persistence
	StartFrame int
	LastFrame  int
	LastLoc    []int
	MeanVec    []float64
	Count      int

	FirstThumb     []byte
	FirstScore     float64
	LastThumb      []byte
	LastScore      float64
	BestThumb      []byte  // Raw JPEG bytes of the best face
	BestQuality    float64 // Best quality score seen so far (Highest)
	LowestThumb    []byte
	LowestScore    float64
	LastSavedScore float64
	PendingFrames  []frameData
	TopFrames      []frameCandidate

	Name        string // Master display name (e.g. "Jenny" or "Identity 1")
	VariantName string // Specific variant name (e.g. "Default", "Glasses")
	IsKnown     bool   // Is this an existing identity from the DB?
}

type identityNameData struct {
	MasterName  string
	VariantName string
}

type frameData struct {
	Index int
	Score float64
	Data  []byte
}

type frameCandidate struct {
	Score float64
	Vec   []float64
	Thumb []byte
}

type timeRange struct {
	Start float64
	End   float64
}

func processResults(ctx context.Context, results <-chan scanResult, db *store.Store, videoID string, fps float64, opts Options, errChan chan<- error, finalIntervalsChan chan<- []store.IntervalData, inflightSem chan struct{}) {
	// Buffer for re-ordering frames (Worker 2 might finish before Worker 1)
	buffer := make(map[int]scanResult)
	nextFrame := 0

	// Holds all final interval data to be batch-inserted at the end
	var finalIntervals []store.IntervalData

	var tracks []*activeTrack

	// Global stats for identities to track extremes across multiple tracks
	globalBestScore := make(map[int]float64)
	globalLowestScore := make(map[int]float64)
	firstDetectionWritten := make(map[int]bool)
	identityDirsCreated := make(map[int]bool)

	variantToMasterID := make(map[int]int)
	idNames := make(map[int]identityNameData)
	newlyCreated := make(map[int]bool) // Track which IDs were generated in this session
	tempIDCounter := -1                // Negative IDs for pending tracks

	summary := make(map[int][]timeRange)
	totalDetections := 0

	var consumerWg sync.WaitGroup
	var producerWg sync.WaitGroup

	gracePeriod, _ := time.ParseDuration(opts.GracePeriod)
	blipDuration, _ := time.ParseDuration(opts.BlipDuration)
	maxGapFrames := int(gracePeriod.Seconds() * fps)
	if maxGapFrames < 1 {
		maxGapFrames = 1 // Ensure at least 1 frame gap to prevent instant closing
	}

	// If /data exists (Docker volume), use it. Otherwise use relative "data" (Local)
	outputBase := "data"
	if _, err := os.Stat("/data"); err == nil {
		outputBase = "/data"
	}
	// Optimization: Ensure output directory exists ONCE, not per-track
	resultsDir := filepath.Join(outputBase, "results", videoID)
	if err := os.MkdirAll(resultsDir, 0755); err != nil {
		utils.ShowError("Failed to create output directory", err, nil)
		select {
		case errChan <- err:
		default:
		}
		return
	}
	fmt.Fprintf(os.Stderr, "📂 Output Directory: %s\n", resultsDir)

	// We use a buffered channel to offload thumbnail disk I/O without blocking the main loop,
	// while ensuring that writes for the same ID happen in order.
	type thumbOp struct {
		dir        string
		filename   string
		data       []byte
		removeGlob string
	}
	// Buffer increased to 1024 to prevent blocking the main loop during heavy track persistence
	thumbChan := make(chan thumbOp, 1024)
	consumerWg.Add(1)
	go func() {
		defer consumerWg.Done()
		for op := range thumbChan {
			if op.removeGlob != "" {
				matches, _ := filepath.Glob(filepath.Join(op.dir, op.removeGlob))
				for _, m := range matches {
					os.Remove(m)
				}
			}

			finalPath := filepath.Join(op.dir, op.filename)
			tempPath := finalPath + ".tmp"

			var err error
			// Write to a temporary file first to prevent corruption
			if err = os.WriteFile(tempPath, op.data, 0644); err == nil {
				// Atomically rename the temp file to its final destination
				if err = os.Rename(tempPath, finalPath); err != nil {
					os.Remove(tempPath) // Clean up temp file on rename failure
				}
			}

			if err != nil {
				fmt.Fprintf(os.Stderr, "⚠️  Failed to save thumbnail %s: %v\n", op.filename, err)
			}
		}
	}()

	// Ensure cleanup happens even if we return early due to error (e.g. DB failure)
	defer func() {
		producerWg.Wait()
		close(thumbChan)
		consumerWg.Wait()
		finalIntervalsChan <- finalIntervals
	}()

	fmt.Fprintf(os.Stderr, "⚙️  Tracker initialized with Cosine Distance (Threshold: %.2f)\n", opts.MatchThreshold)

	// Helper closure to persist a track (DRY: Used in loop and at flush)
	persistTrack := func(t *activeTrack) {
		startSec := float64(t.StartFrame) / fps
		// Include the duration of the last frame slice in the interval
		endSec := float64(t.LastFrame+opts.NthFrame) / fps

		if (endSec - startSec) < blipDuration.Seconds() {
			// Since we use deferred creation (negative IDs), we simply do nothing here
			return
		}

		isNewIdentity := t.ID < 0
		finalVariantID := t.ID // This is the Variant ID

		// Forensic Selection: Select the frame closest to the mean vector from the top candidates
		if len(t.TopFrames) > 0 {
			bestIdx := -1
			minDist := 2.0
			for i, f := range t.TopFrames {
				dist := utils.CosineDist(f.Vec, t.MeanVec)
				if dist < minDist {
					minDist = dist
					bestIdx = i
				}
			}
			if bestIdx != -1 {
				t.BestThumb = t.TopFrames[bestIdx].Thumb
				t.BestQuality = t.TopFrames[bestIdx].Score
			}
		}

		if isNewIdentity {
			// Create Identity Synchronously. If we do this async, a race condition exists
			// where the person reappears before the DB commit, causing a duplicate identity.
			var err error
			var createdMasterID int
			finalVariantID, createdMasterID, err = db.CreateIdentity(ctx, t.MeanVec, t.Count)
			if err != nil {
				utils.ShowError("Failed to create deferred identity", err, nil)
				select {
				case errChan <- err:
				default:
				}
				return
			}
			t.MasterID = createdMasterID
			// Update metadata so the final summary report is correct
			// Note: CreateIdentity creates a Master "Identity <ID>" and Variant "Default"
			// We store the Master Name for display
			variantToMasterID[finalVariantID] = t.MasterID
			idNames[finalVariantID] = identityNameData{MasterName: fmt.Sprintf("Identity %d", t.MasterID), VariantName: "Default"}
			t.ID = finalVariantID               // Update track ID with VariantID
			newlyCreated[finalVariantID] = true // Mark as new for the summary report
		}

		finalMasterID := t.MasterID

		// Create Identity Directory
		identityDir := filepath.Join(resultsDir, fmt.Sprintf("identity_%d", finalMasterID)) // Use MasterID for directory
		framesDir := filepath.Join(identityDir, "frames")
		if !identityDirsCreated[finalMasterID] { // Use MasterID for map key
			if err := os.MkdirAll(framesDir, 0755); err != nil {
				utils.ShowError("Failed to create identity directory", err, nil)
			}
			identityDirsCreated[finalMasterID] = true // Use MasterID for map key
		}

		// 1. First Detection (Only if not written for this ID yet)
		if !firstDetectionWritten[finalMasterID] { // Use MasterID for map key
			thumbChan <- thumbOp{
				dir:        identityDir,
				filename:   fmt.Sprintf("1_First_Detection_[%.2f].jpg", t.FirstScore),
				data:       t.FirstThumb,
				removeGlob: "1_First_Detection_*.jpg",
			}
			firstDetectionWritten[finalMasterID] = true // Use MasterID for map key
		}

		// 2. Last Detection (Always overwrite)
		thumbChan <- thumbOp{
			dir:        identityDir,
			filename:   fmt.Sprintf("2_Last_Detection_[%.2f].jpg", t.LastScore),
			data:       t.LastThumb,
			removeGlob: "2_Last_Detection_*.jpg",
		}

		// 3. Highest Confidence
		if t.BestQuality > globalBestScore[finalMasterID] { // Use MasterID for map key
			globalBestScore[finalMasterID] = t.BestQuality // Use MasterID for map key
			thumbChan <- thumbOp{
				dir:        identityDir,
				filename:   fmt.Sprintf("3_Highest_Confidence_[%.2f].jpg", t.BestQuality),
				data:       t.BestThumb,
				removeGlob: "3_Highest_Confidence_*.jpg",
			}
		}

		// 4. Lowest Confidence
		currLow, ok := globalLowestScore[finalMasterID] // Use MasterID for map key
		if !ok || t.LowestScore < currLow {
			globalLowestScore[finalMasterID] = t.LowestScore // Use MasterID for map key
			thumbChan <- thumbOp{
				dir:        identityDir,
				filename:   fmt.Sprintf("4_Lowest_Confidence_[%.2f].jpg", t.LowestScore),
				data:       t.LowestThumb,
				removeGlob: "4_Lowest_Confidence_*.jpg",
			}
		}

		// Frames (10% change)
		for _, f := range t.PendingFrames {
			thumbChan <- thumbOp{
				dir:      framesDir,
				filename: fmt.Sprintf("frame_[%05d]_score_[%.2f].jpg", f.Index, f.Score),
				data:     f.Data,
			}
		}

		producerWg.Add(1)
		go func(id int, vec []float64, count int, isNew bool) {
			defer producerWg.Done()

			if !isNew {
				// Only update vector if it was already known (accumulate average).
				// For new identities, CreateIdentity already inserted the vector.
				if err := db.UpdateIdentity(ctx, id, vec, count); err != nil { // `id` here is VariantID, correct for UpdateIdentity
					utils.ShowError(fmt.Sprintf("Failed to update variant %d", id), err, nil)
					select {
					case errChan <- err:
					default:
					}
					return
				}
			}

		}(finalVariantID, t.MeanVec, t.Count, isNewIdentity)

		finalIntervals = append(finalIntervals, store.IntervalData{
			Start:     startSec,
			End:       endSec,
			FaceCount: t.Count,
			VariantID: finalVariantID, // This is correct, intervals link to variants
		})
		// Update summary with the final ID (guaranteed to be the correct DB ID)
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

				assignedTracks := make(map[int]bool)

				for _, face := range frame.Faces {
					totalDetections++
					bestMatch := -1
					minDist := opts.MatchThreshold

					for i, t := range tracks {
						if assignedTracks[t.ID] {
							continue
						}

						dist := utils.CosineDist(face.Vec, t.MeanVec)
						if dist < minDist {
							minDist = dist
							bestMatch = i
						}
					}

					if bestMatch != -1 {
						t := tracks[bestMatch]
						t.LastFrame = frame.Index
						t.LastLoc = face.Loc
						assignedTracks[t.ID] = true

						// Update running mean vector
						k := float64(t.Count)
						for j := 0; j < embeddingDim; j++ {
							t.MeanVec[j] = (k*t.MeanVec[j] + face.Vec[j]) / (k + 1.0)
						}
						t.Count++

						t.LastThumb = face.Thumb
						t.LastScore = face.Quality

						if face.Quality > t.BestQuality {
							t.BestQuality = face.Quality
							t.BestThumb = face.Thumb
						}

						// Maintain Top 10 Frames for Centroid Strategy
						candidate := frameCandidate{Score: face.Quality, Vec: face.Vec, Thumb: face.Thumb}
						if len(t.TopFrames) < 10 {
							t.TopFrames = append(t.TopFrames, candidate)
						} else {
							minIdx := -1
							minScore := math.MaxFloat64
							for i, f := range t.TopFrames {
								if f.Score < minScore {
									minScore = f.Score
									minIdx = i
								}
							}
							if face.Quality > minScore {
								t.TopFrames[minIdx] = candidate
							}
						}

						if face.Quality < t.LowestScore {
							t.LowestScore = face.Quality
							t.LowestThumb = face.Thumb
						}

						if math.Abs(face.Quality-t.LastSavedScore)/t.LastSavedScore >= 0.10 {
							t.PendingFrames = append(t.PendingFrames, frameData{Index: frame.Index, Score: face.Quality, Data: face.Thumb})
							t.LastSavedScore = face.Quality
						}
					} else {
						matchVariantID, matchMasterID, matchMasterName, matchVariantName, err := db.FindClosestIdentity(ctx, face.Vec, opts.MatchThreshold)
						if err != nil {
							utils.ShowError("DB Identity Lookup failed", err, nil)
							select {
							case errChan <- err:
							default:
							}
							return
						}

						if matchVariantID != -1 {
							newT := newActiveTrack(matchVariantID, matchMasterID, frame.Index, face.Vec, face.Thumb, face.Quality, matchMasterName, matchVariantName, true, face.Loc)
							tracks = append(tracks, newT)
							variantToMasterID[matchVariantID] = matchMasterID
							idNames[matchVariantID] = identityNameData{MasterName: matchMasterName, VariantName: matchVariantName}
							assignedTracks[matchVariantID] = true
						} else {
							// New Identity -> Use Temporary Negative ID. Defer DB creation until blip filter passes
							tempID := tempIDCounter
							tempIDCounter--

							name := fmt.Sprintf("Identity %d (Pending)", tempID)
							newT := newActiveTrack(tempID, 0, frame.Index, face.Vec, face.Thumb, face.Quality, name, "Default", false, face.Loc)
							tracks = append(tracks, newT)
							// We don't know the master ID yet, so we can't add to variantToMasterID here. It will be added in persistTrack.
							idNames[tempID] = identityNameData{MasterName: name, VariantName: "Default"}
							assignedTracks[tempID] = true

						}
					}
				}

				active := tracks[:0]
				for _, t := range tracks {
					if frame.Index-t.LastFrame > maxGapFrames {
						persistTrack(t)
					} else {
						active = append(active, t)
					}
				}
				tracks = active

				<-inflightSem
				nextFrame += opts.NthFrame
			}
		}
	}

	for _, t := range tracks {
		persistTrack(t)
	}

	fmt.Fprintf(os.Stderr, "\n---------------------------------------------------------\n")
	fmt.Fprintf(os.Stderr, "📊 SCAN SUMMARY\n")
	fmt.Fprintf(os.Stderr, "---------------------------------------------------------\n")

	// Group results by Master Identity
	masterGroups := make(map[int][]int) // MasterID -> []VariantID
	for vid := range summary {
		mid := variantToMasterID[vid]
		masterGroups[mid] = append(masterGroups[mid], vid)
	}

	var masterIDs []int
	for mid := range masterGroups {
		masterIDs = append(masterIDs, mid)
	}
	sort.Ints(masterIDs)

	for _, mid := range masterIDs {
		vids := masterGroups[mid]
		sort.Ints(vids)

		// Derive Master info from the first variant
		firstVID := vids[0]
		names := idNames[firstVID]
		masterName := names.MasterName
		if masterName == "" {
			masterName = fmt.Sprintf("Identity %d", mid)
		}

		thumbNote := ""
		if _, ok := globalBestScore[mid]; ok {
			thumbNote = fmt.Sprintf("(See results/%s/identity_%d/)", videoID, mid)
		}

		// Determine status based on whether any variant is new
		status := "💾"
		for _, vid := range vids {
			if newlyCreated[vid] {
				status = "✨"
				break
			}
		}

		fmt.Fprintf(os.Stderr, "\n👤 %s %s (ID: %d) Found: %s\n", masterName, status, mid, thumbNote)

		for _, vid := range vids {
			vName := idNames[vid].VariantName
			if vName == "" {
				vName = "Default"
			}

			fmt.Fprintf(os.Stderr, "   👉 Variant: %s (ID: %d)\n", vName, vid)
			for _, r := range summary[vid] {
				fmt.Fprintf(os.Stderr, "      %s -> %s\n", fmtTime(r.Start), fmtTime(r.End))
			}
		}
	}

	fmt.Fprintf(os.Stderr, "\n---------------------------------------------------------\n")
	fmt.Fprintf(os.Stderr, "👁️  Total Face Detections:   %d\n", totalDetections)
	fmt.Fprintf(os.Stderr, "---------------------------------------------------------\n")
}

func newActiveTrack(id, masterID, frameIndex int, vec []float64, thumb []byte, quality float64, name, variantName string, isKnown bool, loc []int) *activeTrack {
	t := &activeTrack{
		ID:             id,
		MasterID:       masterID,
		StartFrame:     frameIndex,
		LastFrame:      frameIndex,
		LastLoc:        loc,
		MeanVec:        make([]float64, embeddingDim),
		Count:          1,
		BestThumb:      thumb,
		BestQuality:    quality,
		FirstThumb:     thumb,
		FirstScore:     quality,
		LastThumb:      thumb,
		LastScore:      quality,
		LowestThumb:    thumb,
		LowestScore:    quality,
		LastSavedScore: quality,
		PendingFrames:  []frameData{{Index: frameIndex, Score: quality, Data: thumb}},
		TopFrames:      []frameCandidate{{Score: quality, Vec: vec, Thumb: thumb}},
		Name:           name,
		VariantName:    variantName,
		IsKnown:        isKnown,
	}
	copy(t.MeanVec, vec)
	return t
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
