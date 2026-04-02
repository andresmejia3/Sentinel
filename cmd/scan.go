package cmd

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/andresmejia3/sentinel/internal/store"
	"github.com/andresmejia3/sentinel/internal/types"
	"github.com/andresmejia3/sentinel/internal/utils"
	"github.com/andresmejia3/sentinel/internal/worker"
	"github.com/schollz/progressbar/v3"
	"github.com/spf13/cobra"
	"gopkg.in/yaml.v3"
)

const megabyte = 1024 * 1024
const embeddingDim = 512
const reviewIDBytes = 6
const shortDisplayIDLength = 12

var scanOpts Options
var scanBufferSize int

var scanCmd = &cobra.Command{
	Use:   "scan",
	Short: "Scan video in staging mode by default",
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
	scanCmd.Flags().StringVar(&scanOpts.ReviewFile, "review-file", "", "Custom output path for the staging review YAML (default: data/reviews/<video>.review.yaml)")
	scanCmd.Flags().BoolVar(&scanOpts.NoStaging, "no-staging", false, "Bypass staging mode and write identities and intervals directly to Postgres")

	scanCmd.MarkFlagRequired("input")
	rootCmd.AddCommand(scanCmd)

}

// Buffer pool to reduce GC pressure during scanning
var frameBufferPool = sync.Pool{
	New: func() interface{} { return make([]byte, 0, megabyte) },
}

type scanDB interface {
	FindClosestIdentity(ctx context.Context, vec []float64, threshold float64) (variantID int, identityID int, identityName string, variantName string, err error)
	FindTopIdentities(ctx context.Context, vec []float64, limit int) ([]store.IdentityMatch, error)
	CreateIdentity(ctx context.Context, vec []float64, count int) (variantID int, identityID int, err error)
	UpdateIdentity(ctx context.Context, id int, newVec []float64, newCount int) error
}

// runScan orchestrates the video scanning process: DB setup, Worker Pool, FFmpeg streaming, and Progress tracking.
func runScan(ctx context.Context, opts Options) error {
	// Create a cancellable context so we can stop all workers if one fails
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	// Error channel to catch failures from background goroutines
	errChan := make(chan error, opts.NumEngines+5) // Increased buffer to ensure we never drop critical errors (e.g. Staging failure)

	if err := validateScanFlags(&opts); err != nil {
		return err
	}

	dbOps := scanDB(DB)
	var scanSession *store.ScanSession
	if opts.NoStaging {
		session, err := DB.BeginScanSession(ctx)
		if err != nil {
			return fmt.Errorf("failed to start scan transaction: %w", err)
		}
		scanSession = session
		defer scanSession.Rollback(context.Background())
		dbOps = scanSession
	}

	// Initialize channels early so we can start workers immediately
	taskChan := make(chan types.FrameTask, opts.NumEngines)
	resultsChan := make(chan scanResult, opts.NumEngines*2)
	var wg sync.WaitGroup

	// Ensure we return all buffered frames to the pool if we exit early on error
	defer func() {
		cancel()
		wg.Wait()

	DrainTask:
		for {
			select {
			case t, ok := <-taskChan:
				if !ok {
					break DrainTask
				}
				frameBufferPool.Put(t.Data)
			default:
				break DrainTask
			}
		}
	}()

	// Flow Control: Prevent OOM by limiting in-flight frames.
	// Limits buffering in the aggregator if a worker stalls.
	semSize := scanBufferSize
	if semSize < 1 {
		semSize = 1
	}
	inflightSem := make(chan struct{}, semSize)

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

	fmt.Fprintln(os.Stderr, "🔐 Fingerprinting video for dedupe and interval tracking...")
	videoID, err := utils.GenerateVideoID(opts.InputPath)
	if err != nil {
		return fmt.Errorf("failed to generate video ID: %w", err)
	}
	reviewID := ""
	if !opts.NoStaging {
		reviewID, err = newReviewID()
		if err != nil {
			return err
		}
		if opts.ReviewFile == "" {
			opts.ReviewFile = defaultReviewFilePath(opts.InputPath, videoID, reviewID)
		}
	}
	if opts.NoStaging {
		if err := DB.EnsureVideoMetadata(ctx, videoID, opts.InputPath); err != nil {
			return fmt.Errorf("failed to register video metadata: %w", err)
		}
	}

	fps, err := utils.GetVideoFPS(ctx, opts.InputPath)
	if err != nil {
		return fmt.Errorf("failed to determine video FPS: %w", err)
	}
	if fps <= 0 || math.IsNaN(fps) || math.IsInf(fps, 0) {
		return fmt.Errorf("invalid video FPS: %f (must be > 0)", fps)
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
	barUpdateStop := make(chan struct{})
	barUpdateDone := make(chan struct{})

	// Start Aggregator (Consumer) concurrently to prevent deadlock on resultsChan
	aggDone := make(chan struct{})
	finalIntervalsChan := make(chan []store.IntervalData, 1)
	go func() {
		processResults(ctx, resultsChan, dbOps, videoID, reviewID, fps, opts, errChan, finalIntervalsChan, inflightSem)
		close(aggDone)
	}()

	go func() {
		defer close(barUpdateDone)
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
			case <-barUpdateStop:
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
		return fmt.Errorf("failed to create FFmpeg stdout pipe: %w", err)
	}
	defer ffmpegOut.Close()

	if err := ffmpeg.Start(); err != nil {
		return fmt.Errorf("failed to start FFmpeg: %w", err)
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
		case err := <-errChan:
			frameBufferPool.Put(buf)
			return err
		case <-ctx.Done():
			frameBufferPool.Put(buf)
			return ctx.Err()
		}

		select {
		case taskChan <- types.FrameTask{Index: virtualIndex, Data: buf}:
			sentFrames++
		case err := <-errChan:
			frameBufferPool.Put(buf)
			return err
		case <-ctx.Done():
			frameBufferPool.Put(buf)
			return ctx.Err()
		}
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("frame scanner failed: %w", err)
	}

	ffmpegWaited = true // Mark as waited so defer doesn't run
	if err := ffmpeg.Wait(); err != nil {
		if stderrBuf.Len() > 0 {
			fmt.Fprintf(os.Stderr, "\nFFmpeg Logs:\n%s\n", stderrBuf.String())
		}
		return fmt.Errorf("FFmpeg execution failed: %w", err)
	}

	bar.Finish()
	close(barUpdateStop)
	<-barUpdateDone

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

	// Safety Check: Verify the aggregator finished successfully before committing.
	// This prevents wiping the database if an error (e.g. MkdirAll) occurred early in processResults.
	select {
	case err := <-errChan:
		return err
	default:
	}

	// Review mode is the default. It writes a YAML file and leaves Postgres untouched.
	if !opts.NoStaging {
		// Final check for review file write errors
		select {
		case err := <-errChan:
			return err
		default:
			// Success
			fmt.Fprintf(os.Stderr, "\n📝 Review file generated for video %s at: %s\n", shortDisplayID(videoID), opts.ReviewFile)
			fmt.Fprintf(os.Stderr, "🧠 Review data file generated at: %s\n", reviewDataFilePath(opts.ReviewFile))
			return nil
		}
	}

	// Atomic Commit: All intervals are inserted in a single transaction.
	fmt.Fprintf(os.Stderr, "🗄️  Committing %d intervals to database...\n", len(intervals))
	if scanSession == nil {
		return fmt.Errorf("scan transaction was not initialized")
	}
	if err := scanSession.FinalizeScan(ctx, videoID, intervals); err != nil {
		return fmt.Errorf("failed to commit scan results: %w", err)
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
	readTimeout, err := time.ParseDuration(opts.WorkerTimeout)
	if err != nil {
		// This should have been caught by validateScanFlags, but as a fallback:
		readTimeout = 30 * time.Second
	}
	cfg := worker.ScanConfig{
		Debug:              opts.DebugScreenshots,
		DetectionThreshold: opts.DetectionThreshold,
		ReadTimeout:        readTimeout,
	}
	pyWorker, err := worker.NewPythonScanWorker(ctx, id, cfg) // Fix shadowing: 'worker' package vs variable
	if err != nil {
		select {
		case <-ctx.Done():
			return
		case errChan <- &utils.ContextualError{Context: "Worker Startup Failed", Err: err}:
			return
		}
	}
	defer pyWorker.Close()

	// Signal to the main thread that this worker is ready
	pidChan <- pyWorker.Cmd.Process.Pid
	ready <- true

	for {
		select {
		case <-ctx.Done():
			return
		case task, ok := <-tasks:
			if !ok {
				return
			}
			faces, err := pyWorker.ProcessScanFrame(task.Data)

			// Return buffer to pool immediately after sending
			frameBufferPool.Put(task.Data)

			if err != nil {
				pyWorker.Close() // Reap process before diagnostics so ExitCode is available
				utils.ShowError("Python crashed", err, pyWorker.Cmd)
				select {
				case <-ctx.Done():
					return
				case errChan <- &utils.SilentError{Err: err}:
					return
				}
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

// StagingItem represents a track to be reviewed in YAML.
type StagingItem struct {
	ID                int                `yaml:"id,omitempty"`
	StartTime         float64            `yaml:"start_time"`
	EndTime           float64            `yaml:"end_time"`
	NearestCandidates []NearestCandidate `yaml:"nearest_candidates,omitempty"`
	Confidence        float64            `yaml:"confidence"`
	Reason            string             `yaml:"reason,omitempty"`
	Identity          string             `yaml:"identity"`
	Variant           string             `yaml:"variant"`
	Action            string             `yaml:"action"`
	// Internal data for the commit process
	InternalVector []float64 `yaml:"-"`
	InternalCount  int       `yaml:"-"`
}

type NearestCandidate struct {
	Identity string  `yaml:"identity"`
	Variant  string  `yaml:"variant,omitempty"`
	Distance float64 `yaml:"distance"`
}

type ReviewDocument struct {
	ReviewID     string        `yaml:"review_id,omitempty"`
	VideoID      string        `yaml:"video_id,omitempty"`
	InputPath    string        `yaml:"input_path,omitempty"`
	Tracks       []StagingItem `yaml:"tracks"`
	ArtifactRoot string        `yaml:"-"`
}

type ReviewTrackData struct {
	Fingerprint    string    `json:"fingerprint,omitempty"`
	InternalVector []float64 `json:"internal_vector"`
	InternalCount  int       `json:"internal_count"`
}

type ReviewSidecarDocument struct {
	ReviewID  string                     `json:"review_id,omitempty"`
	VideoID   string                     `json:"video_id,omitempty"`
	InputPath string                     `json:"input_path,omitempty"`
	Tracks    map[string]ReviewTrackData `json:"tracks"`
}

type reviewSummaryItem struct {
	ID                int
	StartTime         float64
	EndTime           float64
	NearestCandidates []NearestCandidate
	Identity          string
	Variant           string
	Confidence        float64
	Reason            string
	Action            string
	KnownVariant      bool
	ArtifactDir       string
}

// --- Aggregation & Tracking Logic ---

type activeTrack struct {
	ID         int
	IdentityID int // Optimization: Store IdentityID to avoid DB lookups during persistence
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

	IdentityName string // Identity display name (e.g. "Jenny" or "Identity 1")
	VariantName  string // Specific variant name (e.g. "Default", "Glasses")
	IsKnown      bool   // Is this an existing identity from the DB?
}

type identityNameData struct {
	IdentityName string
	VariantName  string
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

func stagingItemKey(item StagingItem) string {
	if item.ID > 0 {
		return strconv.Itoa(item.ID)
	}
	return ""
}

func stagingItemLabel(item StagingItem) string {
	if item.ID > 0 {
		return strconv.Itoa(item.ID)
	}
	return "unknown"
}

func reviewDocumentID(doc ReviewDocument) string {
	return doc.ReviewID
}

func sidecarDocumentID(doc ReviewSidecarDocument) string {
	return doc.ReviewID
}

func newReviewID() (string, error) {
	buf := make([]byte, reviewIDBytes)
	if _, err := rand.Read(buf); err != nil {
		return "", fmt.Errorf("failed to generate review_id: %w", err)
	}
	return hex.EncodeToString(buf), nil
}

func shortDisplayID(id string) string {
	if len(id) <= shortDisplayIDLength {
		return id
	}
	return id[:shortDisplayIDLength]
}

func reviewArtifactRelativeDir(videoID, reviewID string, trackID int) string {
	return filepath.Join("results", videoID, "reviews", reviewID, "tracks", strconv.Itoa(trackID))
}

func reviewArtifactsRootDir(resultsDir, reviewID string) string {
	return filepath.Join(resultsDir, "reviews", reviewID)
}

func reviewArtifactDir(resultsDir, reviewID string, trackID int) string {
	return filepath.Join(reviewArtifactsRootDir(resultsDir, reviewID), "tracks", strconv.Itoa(trackID))
}

func reviewArtifactCommentRoot(outputBase, videoID, reviewID string) string {
	if outputBase == "/data" {
		return filepath.Join("results", videoID, "reviews", reviewID, "tracks")
	}
	return filepath.Join(outputBase, "results", videoID, "reviews", reviewID, "tracks")
}

func prepareReviewArtifactsRoot(reviewArtifactsDir string) error {
	if err := os.RemoveAll(reviewArtifactsDir); err != nil {
		return fmt.Errorf("failed to clear review artifacts: %w", err)
	}
	reviewTracksDir := filepath.Join(reviewArtifactsDir, "tracks")
	if err := os.MkdirAll(reviewTracksDir, 0755); err != nil {
		return fmt.Errorf("failed to recreate review artifacts root: %w", err)
	}
	return nil
}

func reviewTrackFingerprint(item StagingItem) (string, error) {
	payload := struct {
		Key               string             `json:"key"`
		StartTime         float64            `json:"start_time"`
		EndTime           float64            `json:"end_time"`
		NearestCandidates []NearestCandidate `json:"nearest_candidates,omitempty"`
		Confidence        float64            `json:"confidence"`
		Reason            string             `json:"reason,omitempty"`
	}{
		Key:               stagingItemKey(item),
		StartTime:         item.StartTime,
		EndTime:           item.EndTime,
		NearestCandidates: item.NearestCandidates,
		Confidence:        item.Confidence,
		Reason:            item.Reason,
	}

	data, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("failed to fingerprint review track %s: %w", stagingItemLabel(item), err)
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), nil
}

func identityVariantLabel(identity, variant string) string {
	switch {
	case identity != "" && variant != "":
		return fmt.Sprintf("%s / %s", identity, variant)
	case identity != "":
		return identity
	default:
		return "none"
	}
}

func roundTo(v float64, places int) float64 {
	if places < 0 {
		return v
	}
	factor := math.Pow(10, float64(places))
	return math.Round(v*factor) / factor
}

func candidateLabel(candidate NearestCandidate) string {
	return identityVariantLabel(candidate.Identity, candidate.Variant)
}

func reviewReason(matches []store.IdentityMatch, matchThreshold float64, action string) string {
	if len(matches) == 0 {
		return "no stored variants found -> suggested new_identity"
	}

	topDistance := roundTo(matches[0].Distance, 3)
	switch action {
	case "new_identity":
		return fmt.Sprintf("nearest_distance=%.3f > threshold=%.3f -> suggested new_identity", topDistance, roundTo(matchThreshold, 3))
	case "":
		if len(matches) > 1 {
			secondDistance := roundTo(matches[1].Distance, 3)
			gap := roundTo(math.Abs(matches[1].Distance-matches[0].Distance), 3)
			return fmt.Sprintf("nearest_distance=%.3f, second_distance=%.3f, gap=%.3f < 0.050 -> needs review", topDistance, secondDistance, gap)
		}
		return fmt.Sprintf("nearest_distance=%.3f -> needs review", topDistance)
	case "new_variant":
		return fmt.Sprintf("nearest_distance=%.3f is within threshold=%.3f but above merge_cutoff=0.350 -> suggested new_variant", topDistance, roundTo(matchThreshold, 3))
	case "merge":
		return fmt.Sprintf("nearest_distance=%.3f <= merge_cutoff=0.350 -> suggested merge", topDistance)
	default:
		return fmt.Sprintf("nearest_distance=%.3f -> suggested %s", topDistance, action)
	}
}

func reviewActionLabel(action string) string {
	switch action {
	case "":
		return "needs_review"
	default:
		return action
	}
}

func printReviewSummary(videoID string, items []reviewSummaryItem, totalDetections int) {
	sort.Slice(items, func(i, j int) bool { return items[i].ID < items[j].ID })

	fmt.Fprintf(os.Stderr, "\n---------------------------------------------------------\n")
	fmt.Fprintf(os.Stderr, "📋 REVIEW SUMMARY [%s]\n", shortDisplayID(videoID))
	fmt.Fprintf(os.Stderr, "---------------------------------------------------------\n")

	for _, item := range items {
		status := "pending track"
		trackingHint := ""
		if item.KnownVariant {
			if item.Action == "merge" {
				status = "Possible known variant"
			} else {
				trackingHint = "possible known variant"
			}
		}

		candidates := item.NearestCandidates
		targetVariant := item.Variant
		if item.Action == "new_variant" && targetVariant == "" && item.Identity != "" {
			targetVariant = "[set during review]"
		}
		target := identityVariantLabel(item.Identity, targetVariant)

		fmt.Fprintf(os.Stderr, "\n🎞️ Track %d [%s]\n", item.ID, status)
		fmt.Fprintf(os.Stderr, "   time: %s -> %s\n", utils.FmtTime(item.StartTime), utils.FmtTime(item.EndTime))
		fmt.Fprintf(os.Stderr, "   action: %s\n", reviewActionLabel(item.Action))
		if trackingHint != "" {
			fmt.Fprintf(os.Stderr, "   tracking hint: %s\n", trackingHint)
		}
		if len(candidates) == 0 {
			fmt.Fprintf(os.Stderr, "   nearest: none\n")
		} else {
			fmt.Fprintf(os.Stderr, "   nearest:\n")
			for i, candidate := range candidates {
				fmt.Fprintf(os.Stderr, "     %d. %s (distance: %.3f)\n", i+1, candidateLabel(candidate), candidate.Distance)
			}
		}
		fmt.Fprintf(os.Stderr, "   target: %s\n", target)
		fmt.Fprintf(os.Stderr, "   confidence: %.2f\n", item.Confidence)
		if item.Reason != "" {
			fmt.Fprintf(os.Stderr, "   reason: %s\n", item.Reason)
		}
		fmt.Fprintf(os.Stderr, "   artifacts: %s/\n", item.ArtifactDir)
	}

	fmt.Fprintf(os.Stderr, "\n---------------------------------------------------------\n")
	fmt.Fprintf(os.Stderr, "👁️  Total Face Detections:   %d\n", totalDetections)
	fmt.Fprintf(os.Stderr, "---------------------------------------------------------\n")
}

func processResults(ctx context.Context, results <-chan scanResult, db scanDB, videoID, reviewID string, fps float64, opts Options, errChan chan<- error, finalIntervalsChan chan<- []store.IntervalData, inflightSem chan struct{}) {
	var reviewFileReadyToWrite bool
	var finalIntervals []store.IntervalData
	var reviewItems []StagingItem
	var reviewSummaryItems []reviewSummaryItem
	var resultsDir string
	var artifactCommentRoot string

	var consumerWg sync.WaitGroup

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

	// Ensure cleanup happens even if we return early due to error (e.g. DB failure or MkdirAll)
	defer func() {
		close(thumbChan)
		consumerWg.Wait()

		// Only write the review file if the process completed successfully.
		if reviewFileReadyToWrite && opts.ReviewFile != "" {
			doc := ReviewDocument{
				ReviewID:     reviewID,
				VideoID:      videoID,
				InputPath:    opts.InputPath,
				Tracks:       reviewItems,
				ArtifactRoot: artifactCommentRoot,
			}
			if err := writeReviewArtifacts(opts.ReviewFile, doc); err != nil {
				// Try to send the error back to the main routine.
				// Use a non-blocking send in case the channel is already full or closed.
				select {
				case errChan <- err:
				default:
				}
			}
		}

		finalIntervalsChan <- finalIntervals
	}()

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

	// 2. Initialize processing variables
	// Buffer for re-ordering frames (Worker 2 might finish before Worker 1)
	buffer := make(map[int]scanResult)
	nextFrame := 0

	var tracks []*activeTrack

	// Global stats for identities to track extremes across multiple tracks
	globalBestScore := make(map[int]float64)
	globalLowestScore := make(map[int]float64)
	firstDetectionWritten := make(map[int]bool)
	identityDirsCreated := make(map[int]bool)

	variantToIdentityID := make(map[int]int)
	idNames := make(map[int]identityNameData)
	newlyCreated := make(map[int]bool) // Track which IDs were generated in this session
	tempIDCounter := -1                // Negative IDs for pending tracks
	nextReviewID := 1

	summary := make(map[int][]timeRange)
	totalDetections := 0

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
	artifactCommentRoot = reviewArtifactCommentRoot(outputBase, videoID, reviewID)
	// Optimization: Ensure output directory exists ONCE, not per-track
	resultsDir = filepath.Join(outputBase, "results", videoID)
	if err := os.MkdirAll(resultsDir, 0755); err != nil {
		sendErr(ctx, errChan, fmt.Errorf("failed to create output directory: %w", err))
		return
	}
	if opts.ReviewFile != "" {
		if err := prepareReviewArtifactsRoot(reviewArtifactsRootDir(resultsDir, reviewID)); err != nil {
			sendErr(ctx, errChan, err)
			return
		}
	}
	fmt.Fprintf(os.Stderr, "📂 Output Directory [%s]: %s\n", shortDisplayID(videoID), resultsDir)

	fmt.Fprintf(os.Stderr, "⚙️  Tracker initialized with Cosine Distance (Threshold: %.2f)\n", opts.MatchThreshold)

	// Helper closure to persist a track (DRY: Used in loop and at flush)
	sendThumbOp := func(op thumbOp) bool {
		// This select is critical to prevent deadlocks on shutdown if the disk is slow
		// and this channel's buffer fills up.
		select {
		case thumbChan <- op:
			return true
		case <-ctx.Done():
			return false
		}
	}

	persistTrack := func(t *activeTrack) {
		startSec := float64(t.StartFrame) / fps
		// Include the duration of the last frame slice in the interval
		endSec := float64(t.LastFrame+opts.NthFrame) / fps

		if (endSec - startSec) < blipDuration.Seconds() {
			// Since we use deferred creation (negative IDs), we simply do nothing here
			if t.ID < 0 {
				delete(idNames, t.ID) // Fix: Prevent memory leak by cleaning up discarded track names
			}
			return
		}

		// --- Review Mode Logic ---
		if opts.ReviewFile != "" {
			trackReviewID := nextReviewID
			nextReviewID++

			trackDir := reviewArtifactDir(resultsDir, reviewID, trackReviewID)
			framesDir := filepath.Join(trackDir, "frames")
			if err := os.MkdirAll(framesDir, 0755); err != nil {
				sendErr(ctx, errChan, fmt.Errorf("failed to create review track directory: %w", err))
				return
			}

			if !sendThumbOp(thumbOp{
				dir:        trackDir,
				filename:   fmt.Sprintf("1_First_Detection_[%.2f].jpg", t.FirstScore),
				data:       t.FirstThumb,
				removeGlob: "1_First_Detection_*.jpg",
			}) {
				return
			}
			if !sendThumbOp(thumbOp{
				dir:        trackDir,
				filename:   fmt.Sprintf("2_Last_Detection_[%.2f].jpg", t.LastScore),
				data:       t.LastThumb,
				removeGlob: "2_Last_Detection_*.jpg",
			}) {
				return
			}
			if !sendThumbOp(thumbOp{
				dir:        trackDir,
				filename:   fmt.Sprintf("3_Highest_Confidence_[%.2f].jpg", t.BestQuality),
				data:       t.BestThumb,
				removeGlob: "3_Highest_Confidence_*.jpg",
			}) {
				return
			}
			if !sendThumbOp(thumbOp{
				dir:        trackDir,
				filename:   fmt.Sprintf("4_Lowest_Confidence_[%.2f].jpg", t.LowestScore),
				data:       t.LowestThumb,
				removeGlob: "4_Lowest_Confidence_*.jpg",
			}) {
				return
			}
			for _, f := range t.PendingFrames {
				if !sendThumbOp(thumbOp{
					dir:      framesDir,
					filename: fmt.Sprintf("frame_[%05d]_score_[%.2f].jpg", f.Index, f.Score),
					data:     f.Data,
				}) {
					return
				}
			}

			// Ranked k-NN Logic
			// 1. Get Top 2 matches (as requested)
			matches, err := db.FindTopIdentities(ctx, t.MeanVec, 2)
			if err != nil {
				sendErr(ctx, errChan, fmt.Errorf("failed to find top identities for staging: %w", err))
				return
			}

			item := StagingItem{
				ID:             trackReviewID,
				StartTime:      startSec,
				EndTime:        endSec,
				InternalVector: t.MeanVec,
				InternalCount:  t.Count,
			}

			if len(matches) > 0 {
				top := matches[0]
				for _, match := range matches {
					item.NearestCandidates = append(item.NearestCandidates, NearestCandidate{
						Identity: match.IdentityName,
						Variant:  match.VariantName,
						Distance: roundTo(match.Distance, 3),
					})
				}
				// Convert distance to confidence (0.0 - 1.0)
				// Assuming simple linear inversion for display: 1.0 - (dist / 2.0)
				// Since cosine dist is 0..2
				conf := 1.0 - (top.Distance / 2.0)
				item.Confidence = math.Round(conf*100) / 100

				// Heuristics
				if top.Distance > opts.MatchThreshold {
					// Too far -> New Identity
					item.Action = "new_identity"
				} else if len(matches) > 1 && math.Abs(matches[1].Distance-top.Distance) < 0.05 {
					// Ambiguous gap -> Leave action blank for review
					item.Action = "" // User must decide
				} else if top.Distance > 0.35 && top.Distance <= opts.MatchThreshold {
					// Logic: It matches (<= Threshold), but it's not super close (> 0.35).
					// This often implies the same person but a different look (sunglasses, beard, etc).
					// Suggest a new variant so we don't pollute the main "Default" cluster.
					item.Action = "new_variant"
				} else {
					// High confidence (<= 0.35) -> Merge
					item.Action = "merge"
				}

				switch item.Action {
				case "merge":
					item.Identity = top.IdentityName
					item.Variant = top.VariantName
				case "new_variant":
					item.Identity = top.IdentityName
					item.Variant = ""
				}
			} else {
				item.Action = "new_identity"
			}
			item.Reason = reviewReason(matches, opts.MatchThreshold, item.Action)

			reviewItems = append(reviewItems, item)
			reviewSummaryItems = append(reviewSummaryItems, reviewSummaryItem{
				ID:                trackReviewID,
				StartTime:         startSec,
				EndTime:           endSec,
				NearestCandidates: item.NearestCandidates,
				Identity:          item.Identity,
				Variant:           item.Variant,
				Confidence:        item.Confidence,
				Reason:            item.Reason,
				Action:            item.Action,
				KnownVariant:      t.IsKnown,
				ArtifactDir:       reviewArtifactRelativeDir(videoID, reviewID, trackReviewID),
			})
			return // Skip DB persistence in review mode
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
			originalTempID := t.ID // Capture the temporary ID before it's updated
			// Create Identity Synchronously. If we do this async, a race condition exists
			// where the person reappears before the DB commit, causing a duplicate identity.
			var err error
			var createdIdentityID int
			finalVariantID, createdIdentityID, err = db.CreateIdentity(ctx, t.MeanVec, t.Count)
			if err != nil {
				sendErr(ctx, errChan, fmt.Errorf("failed to create deferred identity: %w", err))
				return
			}
			t.IdentityID = createdIdentityID
			// Update metadata so the final summary report is correct
			// Note: CreateIdentity creates an Identity "Identity <ID>" and Variant "Default"
			// We store the Identity Name for display
			variantToIdentityID[finalVariantID] = t.IdentityID
			idNames[finalVariantID] = identityNameData{IdentityName: fmt.Sprintf("Identity %d", t.IdentityID), VariantName: "Default"}
			t.ID = finalVariantID               // Update track ID with VariantID
			delete(idNames, originalTempID)     // Fix: Prevent memory leak by cleaning up temporary ID
			newlyCreated[finalVariantID] = true // Mark as new for the summary report
		}

		finalIdentityID := t.IdentityID

		// Create Identity Directory
		identityDir := filepath.Join(resultsDir, fmt.Sprintf("identity_%d", finalIdentityID)) // Use IdentityID for directory
		framesDir := filepath.Join(identityDir, "frames")
		if !identityDirsCreated[finalIdentityID] { // Use IdentityID for map key
			if err := os.MkdirAll(framesDir, 0755); err != nil {
				// Non-fatal, just log warning
				fmt.Fprintf(os.Stderr, "⚠️ Failed to create identity directory: %v\n", err)
			}
			identityDirsCreated[finalIdentityID] = true // Use IdentityID for map key
		}

		// 1. First Detection (Only if not written for this ID yet)
		if !firstDetectionWritten[finalIdentityID] { // Use IdentityID for map key
			if !sendThumbOp(thumbOp{
				dir:        identityDir,
				filename:   fmt.Sprintf("1_First_Detection_[%.2f].jpg", t.FirstScore),
				data:       t.FirstThumb,
				removeGlob: "1_First_Detection_*.jpg",
			}) {
				return
			}
			firstDetectionWritten[finalIdentityID] = true // Use IdentityID for map key
		}

		// 2. Last Detection (Always overwrite)
		if !sendThumbOp(thumbOp{
			dir:        identityDir,
			filename:   fmt.Sprintf("2_Last_Detection_[%.2f].jpg", t.LastScore),
			data:       t.LastThumb,
			removeGlob: "2_Last_Detection_*.jpg",
		}) {
			return
		}

		// 3. Highest Confidence
		if t.BestQuality > globalBestScore[finalIdentityID] { // Use IdentityID for map key
			globalBestScore[finalIdentityID] = t.BestQuality // Use IdentityID for map key
			if !sendThumbOp(thumbOp{
				dir:        identityDir,
				filename:   fmt.Sprintf("3_Highest_Confidence_[%.2f].jpg", t.BestQuality),
				data:       t.BestThumb,
				removeGlob: "3_Highest_Confidence_*.jpg",
			}) {
				return
			}
		}

		// 4. Lowest Confidence
		currLow, ok := globalLowestScore[finalIdentityID] // Use IdentityID for map key
		if !ok || t.LowestScore < currLow {
			globalLowestScore[finalIdentityID] = t.LowestScore // Use IdentityID for map key
			if !sendThumbOp(thumbOp{
				dir:        identityDir,
				filename:   fmt.Sprintf("4_Lowest_Confidence_[%.2f].jpg", t.LowestScore),
				data:       t.LowestThumb,
				removeGlob: "4_Lowest_Confidence_*.jpg",
			}) {
				return
			}
		}

		// Frames (10% change)
		for _, f := range t.PendingFrames {
			if !sendThumbOp(thumbOp{
				dir:      framesDir,
				filename: fmt.Sprintf("frame_[%05d]_score_[%.2f].jpg", f.Index, f.Score),
				data:     f.Data,
			}) {
				return
			}
		}

		if !isNewIdentity {
			// Keep variant math inside the active scan transaction so the whole scan commits atomically.
			if err := db.UpdateIdentity(ctx, finalVariantID, t.MeanVec, t.Count); err != nil {
				sendErr(ctx, errChan, fmt.Errorf("failed to update variant %d: %w", finalVariantID, err))
				return
			}
		}

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

			// Keep frame order strict. The in-flight semaphore already bounds memory,
			// so a missing frame should stall the pipeline rather than be reordered.
			for {
				frame, ok := buffer[nextFrame]
				if !ok {
					break
				}
				delete(buffer, nextFrame)

				aggregateFrameResults(ctx, frame, &tracks, &totalDetections, opts, db, errChan, variantToIdentityID, idNames, &tempIDCounter, maxGapFrames, persistTrack)
				<-inflightSem
				nextFrame += opts.NthFrame
			}
		}
	}

	if len(buffer) > 0 {
		sendErr(ctx, errChan, fmt.Errorf("scan stopped with missing frame %d; %d later result(s) remained buffered", nextFrame, len(buffer)))
		return
	}

	for _, t := range tracks {
		persistTrack(t)
	}

	if opts.ReviewFile != "" {
		printReviewSummary(videoID, reviewSummaryItems, totalDetections)
	} else {
		fmt.Fprintf(os.Stderr, "\n---------------------------------------------------------\n")
		fmt.Fprintf(os.Stderr, "📊 SCAN SUMMARY [%s]\n", shortDisplayID(videoID))
		fmt.Fprintf(os.Stderr, "---------------------------------------------------------\n")

		// Group results by Identity
		identityGroups := make(map[int][]int) // IdentityID -> []VariantID
		for vid := range summary {
			mid := variantToIdentityID[vid]
			identityGroups[mid] = append(identityGroups[mid], vid)
		}

		var identityIDs []int
		for mid := range identityGroups {
			identityIDs = append(identityIDs, mid)
		}
		sort.Ints(identityIDs)

		for _, mid := range identityIDs {
			vids := identityGroups[mid]
			sort.Ints(vids)

			// Derive Identity info from the first variant
			firstVID := vids[0]
			names := idNames[firstVID]
			identityName := names.IdentityName
			if identityName == "" {
				identityName = fmt.Sprintf("Identity %d", mid)
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

			fmt.Fprintf(os.Stderr, "\n👤 %s %s (ID: %d) Found: %s\n", identityName, status, mid, thumbNote)

			for _, vid := range vids {
				vName := idNames[vid].VariantName
				if vName == "" {
					vName = "Default"
				}

				fmt.Fprintf(os.Stderr, "   👉 Variant: %s (ID: %d)\n", vName, vid)
				for _, r := range summary[vid] {
					fmt.Fprintf(os.Stderr, "      %s -> %s\n", utils.FmtTime(r.Start), utils.FmtTime(r.End))
				}
			}
		}

		fmt.Fprintf(os.Stderr, "\n---------------------------------------------------------\n")
		fmt.Fprintf(os.Stderr, "👁️  Total Face Detections:   %d\n", totalDetections)
		fmt.Fprintf(os.Stderr, "---------------------------------------------------------\n")
	}
	reviewFileReadyToWrite = true
}

// aggregateFrameResults encapsulates the core tracking and identity logic for a single frame.
// Extracted to support both the main sequential aggregator and the final buffer flush.
func aggregateFrameResults(ctx context.Context, frame scanResult, tracks *[]*activeTrack, totalDetections *int, opts Options, db scanDB, errChan chan<- error, variantToIdentityID map[int]int, idNames map[int]identityNameData, tempIDCounter *int, maxGapFrames int, persistTrack func(*activeTrack)) {
	assignedTracks := make(map[int]bool)
	isTracking := make(map[int]bool)

	// Enforce the grace period before matching the current frame.
	// Once a track is stale, it should no longer be eligible to absorb new detections.
	active := (*tracks)[:0]
	for _, t := range *tracks {
		if frame.Index-t.LastFrame > maxGapFrames {
			persistTrack(t)
			continue
		}
		active = append(active, t)
		isTracking[t.ID] = true
	}
	*tracks = active

	startPendingTrack := func(face types.FaceResult) {
		tempID := *tempIDCounter
		*tempIDCounter--
		name := fmt.Sprintf("Identity %d (Pending)", tempID)
		newT := newActiveTrack(tempID, 0, frame.Index, face.Vec, face.Thumb, face.Quality, name, "Default", false, face.Loc)
		*tracks = append(*tracks, newT)
		isTracking[tempID], assignedTracks[tempID] = true, true
		idNames[tempID] = identityNameData{IdentityName: name, VariantName: "Default"}
	}

	assignFaceToTrack := func(face types.FaceResult, t *activeTrack) {
		t.LastFrame = frame.Index
		t.LastLoc = face.Loc
		assignedTracks[t.ID] = true

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

		candidate := frameCandidate{Score: face.Quality, Vec: face.Vec, Thumb: face.Thumb}
		if len(t.TopFrames) < 10 {
			t.TopFrames = append(t.TopFrames, candidate)
		} else {
			minIdx, minScore := -1, math.MaxFloat64
			for i, f := range t.TopFrames {
				if f.Score < minScore {
					minScore, minIdx = f.Score, i
				}
			}
			if face.Quality > minScore {
				t.TopFrames[minIdx] = candidate
			}
		}
		if face.Quality < t.LowestScore {
			t.LowestScore, t.LowestThumb = face.Quality, face.Thumb
		}

		// Logic Safety: Use a single robust check for sample frame capture
		if t.LastSavedScore > 0.001 && math.Abs(face.Quality-t.LastSavedScore)/t.LastSavedScore >= 0.10 {
			t.PendingFrames = append(t.PendingFrames, frameData{Index: frame.Index, Score: face.Quality, Data: face.Thumb})
			t.LastSavedScore = face.Quality
		}
	}

	type activeProposal struct {
		faceIdx  int
		trackIdx int
		dist     float64
	}

	// Build all active-track proposals first so same-frame assignment is not affected
	// by the order the detector happened to return faces in.
	proposalsByTrack := make(map[int][]activeProposal)
	hasActiveProposal := make([]bool, len(frame.Faces))
	conflictedFace := make([]bool, len(frame.Faces))
	assignedByActive := make([]bool, len(frame.Faces))

	for faceIdx, face := range frame.Faces {
		*totalDetections++
		bestMatch := -1
		minDist := opts.MatchThreshold

		for i, t := range *tracks {
			dist := utils.CosineDist(face.Vec, t.MeanVec)
			if dist < minDist {
				minDist = dist
				bestMatch = i
			}
		}

		if bestMatch != -1 {
			hasActiveProposal[faceIdx] = true
			proposalsByTrack[bestMatch] = append(proposalsByTrack[bestMatch], activeProposal{
				faceIdx:  faceIdx,
				trackIdx: bestMatch,
				dist:     minDist,
			})
		}
	}

	// Resolve conflicts conservatively: the strongest claimant keeps the track,
	// everyone else becomes a pending track for human review rather than risking
	// a wrong second-choice auto-merge.
	for trackIdx, proposals := range proposalsByTrack {
		winner := proposals[0]
		for _, proposal := range proposals[1:] {
			if proposal.dist < winner.dist {
				winner = proposal
			}
		}

		assignFaceToTrack(frame.Faces[winner.faceIdx], (*tracks)[trackIdx])
		assignedByActive[winner.faceIdx] = true

		for _, proposal := range proposals {
			if proposal.faceIdx == winner.faceIdx {
				continue
			}
			conflictedFace[proposal.faceIdx] = true
		}
	}

	for faceIdx, face := range frame.Faces {
		if assignedByActive[faceIdx] {
			continue
		}

		if conflictedFace[faceIdx] {
			startPendingTrack(face)
			continue
		}

		if !hasActiveProposal[faceIdx] {
			matchVariantID, matchIdentityID, matchIdentityName, matchVariantName, err := db.FindClosestIdentity(ctx, face.Vec, opts.MatchThreshold)
			if err != nil {
				sendErr(ctx, errChan, fmt.Errorf("DB identity lookup failed: %w", err))
				return
			}

			if matchVariantID != -1 {
				if isTracking[matchVariantID] {
					startPendingTrack(face)
					continue
				}
				newT := newActiveTrack(matchVariantID, matchIdentityID, frame.Index, face.Vec, face.Thumb, face.Quality, matchIdentityName, matchVariantName, true, face.Loc)
				*tracks = append(*tracks, newT)
				variantToIdentityID[matchVariantID], idNames[matchVariantID] = matchIdentityID, identityNameData{matchIdentityName, matchVariantName}
				isTracking[matchVariantID], assignedTracks[matchVariantID] = true, true
			} else {
				startPendingTrack(face)
			}
		}
	}
}

// sendErr is a helper to perform a context-aware blocking send on an error channel.
func sendErr(ctx context.Context, errChan chan<- error, err error) {
	select {
	case errChan <- err:
	case <-ctx.Done():
	}
}

func writeReviewArtifacts(path string, doc ReviewDocument) error {
	reviewDoc, reviewData, err := splitReviewDocument(doc)
	if err != nil {
		return err
	}
	if err := writeReviewFile(path, reviewDoc); err != nil {
		return err
	}
	if err := writeReviewSidecarFile(reviewDataFilePath(path), reviewData); err != nil {
		return err
	}
	return nil
}

func splitReviewDocument(doc ReviewDocument) (ReviewDocument, ReviewSidecarDocument, error) {
	reviewDoc := ReviewDocument{
		ReviewID:     reviewDocumentID(doc),
		VideoID:      doc.VideoID,
		InputPath:    doc.InputPath,
		Tracks:       make([]StagingItem, 0, len(doc.Tracks)),
		ArtifactRoot: doc.ArtifactRoot,
	}
	reviewData := ReviewSidecarDocument{
		ReviewID:  reviewDocumentID(doc),
		VideoID:   doc.VideoID,
		InputPath: doc.InputPath,
		Tracks:    make(map[string]ReviewTrackData, len(doc.Tracks)),
	}

		for _, item := range doc.Tracks {
			sanitized := item
			sanitized.InternalVector = nil
			sanitized.InternalCount = 0
			reviewDoc.Tracks = append(reviewDoc.Tracks, sanitized)

		if len(item.InternalVector) == 0 && item.InternalCount == 0 {
			continue
		}
		key := stagingItemKey(item)
		if key == "" {
			return ReviewDocument{}, ReviewSidecarDocument{}, fmt.Errorf("cannot write review data for track with blank id")
		}
		fingerprint, err := reviewTrackFingerprint(item)
		if err != nil {
			return ReviewDocument{}, ReviewSidecarDocument{}, err
		}
		reviewData.Tracks[key] = ReviewTrackData{
			Fingerprint:    fingerprint,
			InternalVector: item.InternalVector,
			InternalCount:  item.InternalCount,
		}
	}

	return reviewDoc, reviewData, nil
}

func writeReviewFile(path string, doc ReviewDocument) error {
	dir := filepath.Dir(path)
	if dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return fmt.Errorf("failed to create review directory: %w", err)
		}
	}
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create review file: %w", err)
	}
	defer f.Close()
	var buf bytes.Buffer
	enc := yaml.NewEncoder(&buf)
	enc.SetIndent(4)
	if err := enc.Encode(doc); err != nil {
		return fmt.Errorf("failed to encode review YAML: %w", err)
	}
	if err := enc.Close(); err != nil {
		return fmt.Errorf("failed to finalize review YAML: %w", err)
	}
	formatted := strings.TrimPrefix(buf.String(), "---\n")
	header := "# Edit only `identity`, `variant`, and `action`.\n" +
		"# `review_id` is the unique identifier for this scan run and review file.\n" +
		"# Leave `review_id`, `video_id`, and `input_path` unchanged.\n" +
		"# Leave `id`, `start_time`, `end_time`, `nearest_candidates`, `confidence`, and `reason` unchanged.\n" +
		"# Valid actions: merge | new_identity | new_variant | discard.\n" +
		"# The prefilled `action` is Sentinel's best guess from scan heuristics, not final truth.\n"
	if doc.ArtifactRoot != "" {
		header += "# Corresponding thumbnails and frame artifacts live under:\n"
		header += fmt.Sprintf("#   %s/<id>/\n", doc.ArtifactRoot)
	}
	formatted = header + formatted
	formatted = strings.ReplaceAll(formatted, "\n    - id:", "\n\n    # Read-only fields below are generated by scan.\n    - id:")
	formatted = strings.ReplaceAll(formatted, "\n      identity:", "\n      # Editable fields below are the values that will be committed.\n      # For `new_identity`, if `identity` is left blank, then Sentinel will auto-name it. If `variant` is left blank, then Sentinel will create the `Default` variant.\n      # For `new_variant`, set `identity` to the existing person and `variant` to the new variant name.\n      identity:")
	if _, err := f.WriteString(formatted); err != nil {
		return fmt.Errorf("failed to write review YAML: %w", err)
	}
	return nil
}

func writeReviewSidecarFile(path string, doc ReviewSidecarDocument) error {
	dir := filepath.Dir(path)
	if dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return fmt.Errorf("failed to create review data directory: %w", err)
		}
	}
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create review data file: %w", err)
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	if err := enc.Encode(doc); err != nil {
		return fmt.Errorf("failed to encode review data JSON: %w", err)
	}
	return nil
}

func readReviewSidecarFile(path string) (ReviewSidecarDocument, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return ReviewSidecarDocument{}, err
	}
	var doc ReviewSidecarDocument
	if err := json.Unmarshal(data, &doc); err != nil {
		return ReviewSidecarDocument{}, fmt.Errorf("failed to parse review data JSON: %w", err)
	}
	if doc.Tracks == nil {
		doc.Tracks = make(map[string]ReviewTrackData)
	}
	return doc, nil
}

func reviewDataFilePath(reviewPath string) string {
	ext := filepath.Ext(reviewPath)
	switch ext {
	case ".yaml", ".yml":
		return strings.TrimSuffix(reviewPath, ext) + ".data.json"
	default:
		return reviewPath + ".data.json"
	}
}

func newActiveTrack(id, identityID, frameIndex int, vec []float64, thumb []byte, quality float64, name, variantName string, isKnown bool, loc []int) *activeTrack {
	t := &activeTrack{
		ID:             id,
		IdentityID:     identityID,
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
		IdentityName:   name,
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
			return fmt.Errorf("input file does not exist: %w", err)
		}
		return fmt.Errorf("unable to access input file: %w", err)
	}
	if info.IsDir() {
		return fmt.Errorf("input path is a directory, not a video file")
	}
	if opts.NthFrame < 1 {
		return fmt.Errorf("invalid nth-frame interval: must be >= 1, got %d", opts.NthFrame)
	}
	if opts.NumEngines < 1 {
		opts.NumEngines = 1
	}
	if opts.MatchThreshold < 0 || opts.MatchThreshold > 1.0 {
		return fmt.Errorf("invalid match threshold: must be between 0.0 and 1.0, got %f", opts.MatchThreshold)
	}
	if opts.NoStaging && opts.ReviewFile != "" {
		return fmt.Errorf("--no-staging cannot be combined with --review-file")
	}
	if opts.DetectionThreshold < 0 || opts.DetectionThreshold > 1.0 {
		return fmt.Errorf("invalid detection threshold: must be between 0.0 and 1.0, got %f", opts.DetectionThreshold)
	}
	if d, err := time.ParseDuration(opts.BlipDuration); err != nil {
		return fmt.Errorf("invalid blip-duration format: %w (use '100ms', '1s')", err)
	} else if d <= 0 {
		return fmt.Errorf("blip-duration must be positive")
	}
	if d, err := time.ParseDuration(opts.GracePeriod); err != nil {
		return fmt.Errorf("invalid grace-period format: %w (use '2s', '500ms')", err)
	} else if d <= 0 {
		return fmt.Errorf("grace-period must be positive")
	}
	if d, err := time.ParseDuration(opts.WorkerTimeout); err != nil {
		return fmt.Errorf("invalid worker-timeout format: %w (use '30s', '1m')", err)
	} else if d <= 0 {
		return fmt.Errorf("worker-timeout must be positive")
	}

	return nil
}

func defaultReviewFilePath(inputPath, videoID, reviewID string) string {
	outputBase := "data"
	if _, err := os.Stat("/data"); err == nil {
		outputBase = "/data"
	}

	base := filepath.Base(inputPath)
	ext := filepath.Ext(base)
	name := strings.TrimSuffix(base, ext)
	if name == "" {
		name = base
	}
	if name == "" {
		name = "scan"
	}

	filename := fmt.Sprintf("%s.%s.%s.review.yaml", name, shortDisplayID(videoID), reviewID)
	return filepath.Join(outputBase, "reviews", filename)
}
