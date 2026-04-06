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
const potentialIdentityThreshold = 0.35
const potentialIdentityStrongMatchThreshold = 0.25
const potentialIdentityAmbiguityMargin = 0.05
const potentialIdentityMoveMargin = 0.03
const potentialIdentityRefinementMaxRounds = 3

var scanOpts Options
var scanBufferSize int

var scanCmd = &cobra.Command{
	Use:   "scan",
	Short: "Scan a video and stage grouped review artifacts by default",
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
	scanCmd.Flags().BoolVarP(&scanOpts.DebugScreenshots, "debug-screenshots", "d", false, "Save debug images with bounding boxes to debug_frames/ (or data/debug_frames/ outside Docker)")
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

	// Start Aggregator (Consumer) concurrently to prevent deadlock on resultsChan
	aggDone := make(chan struct{})
	finalIntervalsChan := make(chan []store.IntervalData, 1)
	startupReady := make(chan struct{})
	go func() {
		processResults(ctx, resultsChan, dbOps, videoID, reviewID, fps, opts, errChan, finalIntervalsChan, inflightSem, startupReady)
		close(aggDone)
	}()

	select {
	case <-startupReady:
	case err := <-errChan:
		return err
	case <-ctx.Done():
		return ctx.Err()
	}

	bar := progressbar.NewOptions(totalVideoFrames,
		progressbar.OptionSetDescription("🔍 Scanning"),
		progressbar.OptionSetWriter(os.Stderr), // Write bar to Stderr
		progressbar.OptionShowCount(),
	)
	barUpdateStop := make(chan struct{})
	barUpdateDone := make(chan struct{})

	go func() {
		defer close(barUpdateDone)
		updateDesc := func() {
			mem := atomic.LoadUint64(&currentMemory)
			bufLen := len(inflightSem)
			bufCap := cap(inflightSem)
			if mem > 0 {
				bar.Describe(fmt.Sprintf("🔍 Scanning (RAM: %.2f GB | Buffer: %d/%d)", float64(mem)/(1024*1024*1024), bufLen, bufCap))
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
			fmt.Fprintf(os.Stderr, "\n🆔 Session ID: %s\n", reviewID)
			fmt.Fprintf(os.Stderr, "🖼️  Results folder: %s\n", reviewArtifactCommentRoot(currentOutputBase(), videoID, reviewID))
			fmt.Fprintf(os.Stderr, "📝 Review file generated for video %s at: %s\n", shortDisplayID(videoID), userFacingOutputPath(opts.ReviewFile))
			fmt.Fprintf(os.Stderr, "🧠 Review data file generated at: %s\n", userFacingOutputPath(reviewDataFilePath(opts.ReviewFile)))
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
	Identity          string             `yaml:"identity"`
	Variant           string             `yaml:"variant"`
	Action            string             `yaml:"action"`
	GroupID           int                `yaml:"-"`
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
	ReviewID     string              `yaml:"review_id,omitempty"`
	VideoID      string              `yaml:"video_id,omitempty"`
	InputPath    string              `yaml:"input_path,omitempty"`
	Tracks       []StagingItem       `yaml:"tracks"`
	ArtifactRoot string              `yaml:"-"`
	SummaryItems []reviewSummaryItem `yaml:"-"`
}

type ReviewFileDocument struct {
	ReviewID            string                        `yaml:"review_id,omitempty"`
	VideoID             string                        `yaml:"video_id,omitempty"`
	InputPath           string                        `yaml:"input_path,omitempty"`
	RawTracks           map[string]ReviewRawTrackItem `yaml:"raw_tracks"`
	UnresolvedTracks    []string                      `yaml:"unresolved_tracks,omitempty"`
	PotentialIdentities []ReviewPotentialIdentityItem `yaml:"potential_identities"`
	ArtifactRoot        string                        `yaml:"-"`
}

type ReviewRawTrackItem struct {
	StartTime         float64            `yaml:"start_time"`
	EndTime           float64            `yaml:"end_time"`
	NearestCandidates []NearestCandidate `yaml:"nearest_candidates,omitempty"`
	Confidence        float64            `yaml:"confidence"`
	SuggestedIdentity string             `yaml:"suggested_identity"`
	SuggestedVariant  string             `yaml:"suggested_variant"`
	SuggestedAction   string             `yaml:"suggested_action"`
	ArtifactDir       string             `yaml:"artifact_dir,omitempty"`
}

type ReviewPotentialIdentityItem struct {
	ID       int      `yaml:"id"`
	Tracks   []string `yaml:"tracks"`
	Identity string   `yaml:"identity"`
	Variant  string   `yaml:"variant"`
	Action   string   `yaml:"action"`
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
	Action            string
	ArtifactDir       string
	Vector            []float64
	Count             int
}

type PotentialIdentityStatus string

const (
	potentialIdentityStatusNew       PotentialIdentityStatus = "NEW"
	potentialIdentityStatusStrong    PotentialIdentityStatus = "STRONG"
	potentialIdentityStatusPossible  PotentialIdentityStatus = "POSSIBLE"
	potentialIdentityStatusAmbiguous PotentialIdentityStatus = "AMBIGUOUS"
)

type PotentialIdentity struct {
	ID         int
	Members    []PotentialIdentityMember
	SumVec     []float64
	Centroid   []float64
	TotalCount int
}

type PotentialIdentityMember struct {
	Item reviewSummaryItem
	Link PotentialIdentityLink
}

type PotentialIdentityLink struct {
	Status                            PotentialIdentityStatus
	BestPotentialIdentityID           int
	BestPotentialIdentityScore        float64
	BestMemberTrackID                 int
	BestMemberDistance                float64
	SecondBestPotentialIdentityID     int
	SecondBestPotentialIdentityScore  float64
	OverlapBlockedPotentialIdentityID int
	OverlapBlockedTrackID             int
	OverlapBlockedDistance            float64
}

type potentialIdentityCandidate struct {
	IdentityIndex         int
	IdentityID            int
	Score                 float64
	CentroidDistance      float64
	BestMemberDistance    float64
	BestMemberTrackID     int
	OverlapBlockedTrackID int
}

type potentialIdentityDecisionKind string

const (
	potentialIdentityDecisionStay       potentialIdentityDecisionKind = "stay"
	potentialIdentityDecisionMove       potentialIdentityDecisionKind = "move"
	potentialIdentityDecisionUnresolved potentialIdentityDecisionKind = "unresolved"
	potentialIdentityDecisionAttach     potentialIdentityDecisionKind = "attach"
)

type potentialIdentityDecision struct {
	TrackID            int
	Kind               potentialIdentityDecisionKind
	OriginalIdentityID int
	TargetIdentityID   int
	Link               PotentialIdentityLink
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
	BestFrame      int
	LowestThumb    []byte
	LowestScore    float64
	LowestFrame    int
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
	Index int
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

func reviewSummaryItemsFromTracks(tracks []StagingItem) []reviewSummaryItem {
	items := make([]reviewSummaryItem, 0, len(tracks))
	for _, track := range tracks {
		items = append(items, reviewSummaryItem{
			ID:                track.ID,
			StartTime:         track.StartTime,
			EndTime:           track.EndTime,
			NearestCandidates: append([]NearestCandidate(nil), track.NearestCandidates...),
			Identity:          track.Identity,
			Variant:           track.Variant,
			Confidence:        track.Confidence,
			Action:            track.Action,
			Vector:            cloneVector(track.InternalVector),
			Count:             track.InternalCount,
		})
	}
	return items
}

func buildReviewRawTrackItems(summaryItems []reviewSummaryItem) map[string]ReviewRawTrackItem {
	rawTracks := make(map[string]ReviewRawTrackItem, len(summaryItems))
	for _, item := range summaryItems {
		if item.ID <= 0 {
			continue
		}
		rawTracks[strconv.Itoa(item.ID)] = ReviewRawTrackItem{
			StartTime:         item.StartTime,
			EndTime:           item.EndTime,
			NearestCandidates: append([]NearestCandidate(nil), item.NearestCandidates...),
			Confidence:        item.Confidence,
			SuggestedIdentity: item.Identity,
			SuggestedVariant:  item.Variant,
			SuggestedAction:   item.Action,
			ArtifactDir:       item.ArtifactDir,
		}
	}
	return rawTracks
}

func allMembersAgreeOnIdentity(members []PotentialIdentityMember) (string, bool) {
	if len(members) == 0 {
		return "", false
	}
	identity := strings.TrimSpace(members[0].Item.Identity)
	if identity == "" {
		return "", false
	}
	for _, member := range members[1:] {
		if strings.TrimSpace(member.Item.Identity) != identity {
			return "", false
		}
	}
	return identity, true
}

func allMembersAgreeOnAction(members []PotentialIdentityMember) (string, bool) {
	if len(members) == 0 {
		return "", false
	}
	action := strings.TrimSpace(members[0].Item.Action)
	if action == "" {
		return "", false
	}
	for _, member := range members[1:] {
		if strings.TrimSpace(member.Item.Action) != action {
			return "", false
		}
	}
	return action, true
}

func allMembersAgreeOnVariant(members []PotentialIdentityMember) (string, bool) {
	if len(members) == 0 {
		return "", false
	}
	variant := strings.TrimSpace(members[0].Item.Variant)
	if variant == "" {
		return "", false
	}
	for _, member := range members[1:] {
		if strings.TrimSpace(member.Item.Variant) != variant {
			return "", false
		}
	}
	return variant, true
}

func suggestPotentialIdentityReviewFields(members []PotentialIdentityMember) (string, string, string) {
	action, actionOK := allMembersAgreeOnAction(members)
	identity, identityOK := allMembersAgreeOnIdentity(members)
	variant, variantOK := allMembersAgreeOnVariant(members)

	if actionOK {
		switch action {
		case "new_identity":
			return "", "", "new_identity"
		case "discard":
			return "", "", "discard"
		case "new_variant":
			if identityOK {
				return identity, "", "new_variant"
			}
		case "merge":
			if identityOK && variantOK {
				return identity, variant, "merge"
			}
		}
	}

	if identityOK {
		return identity, "", ""
	}

	return "", "", ""
}

func reviewPotentialIdentityItem(id int, members []PotentialIdentityMember) ReviewPotentialIdentityItem {
	item := ReviewPotentialIdentityItem{
		ID:     id,
		Tracks: make([]string, 0, len(members)),
	}
	for _, member := range members {
		item.Tracks = append(item.Tracks, strconv.Itoa(member.Item.ID))
	}
	item.Identity, item.Variant, item.Action = suggestPotentialIdentityReviewFields(members)
	return item
}

func buildReviewUnresolvedTrackRefs(unresolved []PotentialIdentityMember) []string {
	refs := make([]string, 0, len(unresolved))
	for _, member := range unresolved {
		if member.Item.ID <= 0 {
			continue
		}
		refs = append(refs, strconv.Itoa(member.Item.ID))
	}
	return refs
}

func buildReviewFileDocument(doc ReviewDocument) ReviewFileDocument {
	summaryItems := append([]reviewSummaryItem(nil), doc.SummaryItems...)
	if len(summaryItems) == 0 {
		summaryItems = reviewSummaryItemsFromTracks(doc.Tracks)
	}

	identities, unresolved := buildPotentialIdentities(summaryItems)
	fileDoc := ReviewFileDocument{
		ReviewID:            reviewDocumentID(doc),
		VideoID:             doc.VideoID,
		InputPath:           doc.InputPath,
		RawTracks:           buildReviewRawTrackItems(summaryItems),
		UnresolvedTracks:    buildReviewUnresolvedTrackRefs(unresolved),
		PotentialIdentities: make([]ReviewPotentialIdentityItem, 0, len(identities)),
		ArtifactRoot:        doc.ArtifactRoot,
	}

	nextID := 1
	for _, identity := range identities {
		fileDoc.PotentialIdentities = append(fileDoc.PotentialIdentities, reviewPotentialIdentityItem(nextID, identity.Members))
		nextID++
	}
	return fileDoc
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

func currentOutputBase() string {
	outputBase := "data"
	if _, err := os.Stat("/data"); err == nil {
		outputBase = "/data"
	}
	return outputBase
}

func headlineArtifactFilename(order int, label string, frameIndex int, score float64) string {
	return fmt.Sprintf("%d_%s_[frame_%05d]_[%.2f].jpg", order, label, frameIndex, score)
}

func userFacingOutputPath(path string) string {
	clean := filepath.Clean(path)
	if clean == "/data" {
		return "."
	}
	prefix := "/data" + string(os.PathSeparator)
	if strings.HasPrefix(clean, prefix) {
		return strings.TrimPrefix(clean, prefix)
	}
	return clean
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
	}{
		Key:               stagingItemKey(item),
		StartTime:         item.StartTime,
		EndTime:           item.EndTime,
		NearestCandidates: item.NearestCandidates,
		Confidence:        item.Confidence,
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

func reviewActionLabel(action string) string {
	switch action {
	case "":
		return "needs_review"
	default:
		return action
	}
}

func cloneVector(vec []float64) []float64 {
	return append([]float64(nil), vec...)
}

func effectiveTrackCount(item reviewSummaryItem) int {
	if item.Count > 0 {
		return item.Count
	}
	return 1
}

func weightedTrackVector(item reviewSummaryItem) []float64 {
	weighted := cloneVector(item.Vector)
	weight := float64(effectiveTrackCount(item))
	for i := range weighted {
		weighted[i] *= weight
	}
	return weighted
}

func recomputePotentialIdentityCentroid(identity *PotentialIdentity) {
	if identity.TotalCount <= 0 || len(identity.SumVec) == 0 {
		identity.SumVec = nil
		identity.Centroid = nil
		identity.TotalCount = 0
		return
	}

	identity.Centroid = make([]float64, len(identity.SumVec))
	for i := range identity.SumVec {
		identity.Centroid[i] = identity.SumVec[i] / float64(identity.TotalCount)
	}
}

func recomputePotentialIdentityStatsFromMembers(identity *PotentialIdentity) {
	if len(identity.Members) == 0 {
		identity.SumVec = nil
		identity.Centroid = nil
		identity.TotalCount = 0
		return
	}

	var sumVec []float64
	totalCount := 0
	for _, member := range identity.Members {
		weightedVec := weightedTrackVector(member.Item)
		if len(sumVec) == 0 {
			sumVec = make([]float64, len(weightedVec))
		}
		if len(sumVec) != len(weightedVec) {
			continue
		}
		for i := range weightedVec {
			sumVec[i] += weightedVec[i]
		}
		totalCount += effectiveTrackCount(member.Item)
	}

	identity.SumVec = sumVec
	identity.TotalCount = totalCount
	recomputePotentialIdentityCentroid(identity)
}

func recomputeAllPotentialIdentityStatsFromMembers(identities []PotentialIdentity) {
	for i := range identities {
		recomputePotentialIdentityStatsFromMembers(&identities[i])
	}
}

func newPotentialIdentityWithID(id int, item reviewSummaryItem, link PotentialIdentityLink) PotentialIdentity {
	sumVec := weightedTrackVector(item)
	return PotentialIdentity{
		ID: id,
		Members: []PotentialIdentityMember{
			{
				Item: item,
				Link: link,
			},
		},
		SumVec:     sumVec,
		Centroid:   cloneVector(item.Vector),
		TotalCount: effectiveTrackCount(item),
	}
}

func newPotentialIdentity(id int, item reviewSummaryItem, link PotentialIdentityLink) PotentialIdentity {
	return newPotentialIdentityWithID(id, item, link)
}

func addPotentialIdentityMember(identity *PotentialIdentity, item reviewSummaryItem, link PotentialIdentityLink) {
	weight := effectiveTrackCount(item)
	weightedVec := weightedTrackVector(item)
	if len(identity.SumVec) == 0 {
		identity.SumVec = make([]float64, len(weightedVec))
	}
	if len(identity.SumVec) == len(weightedVec) {
		for i := range weightedVec {
			identity.SumVec[i] += weightedVec[i]
		}
	}
	identity.TotalCount += weight
	recomputePotentialIdentityCentroid(identity)
	identity.Members = append(identity.Members, PotentialIdentityMember{
		Item: item,
		Link: link,
	})
}

func removePotentialIdentityMember(identity *PotentialIdentity, memberIdx int) PotentialIdentityMember {
	member := identity.Members[memberIdx]
	weight := effectiveTrackCount(member.Item)
	weightedVec := weightedTrackVector(member.Item)
	if len(identity.SumVec) == len(weightedVec) {
		for i := range weightedVec {
			identity.SumVec[i] -= weightedVec[i]
		}
	}
	identity.TotalCount -= weight
	identity.Members = append(identity.Members[:memberIdx], identity.Members[memberIdx+1:]...)
	recomputePotentialIdentityCentroid(identity)
	return member
}

func trackTimesOverlap(a, b reviewSummaryItem) bool {
	return a.StartTime < b.EndTime && b.StartTime < a.EndTime
}

func potentialIdentityCandidateForItem(item reviewSummaryItem, identity PotentialIdentity, identityIndex int) potentialIdentityCandidate {
	candidate := potentialIdentityCandidate{
		IdentityIndex: identityIndex,
		IdentityID:    identity.ID,
	}
	if len(item.Vector) == 0 || len(identity.Centroid) == 0 {
		candidate.Score = math.Inf(1)
		candidate.CentroidDistance = math.Inf(1)
		candidate.BestMemberDistance = math.Inf(1)
		return candidate
	}

	candidate.CentroidDistance = utils.CosineDist(item.Vector, identity.Centroid)
	candidate.BestMemberDistance = math.Inf(1)

	for _, member := range identity.Members {
		dist := utils.CosineDist(item.Vector, member.Item.Vector)
		if dist < candidate.BestMemberDistance {
			candidate.BestMemberDistance = dist
			candidate.BestMemberTrackID = member.Item.ID
		}
		if candidate.OverlapBlockedTrackID == 0 && trackTimesOverlap(item, member.Item) {
			candidate.OverlapBlockedTrackID = member.Item.ID
		}
	}

	candidate.Score = math.Max(candidate.CentroidDistance, candidate.BestMemberDistance)
	return candidate
}

func sortPotentialIdentityCandidates(candidates []potentialIdentityCandidate) {
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].Score != candidates[j].Score {
			return candidates[i].Score < candidates[j].Score
		}
		if candidates[i].BestMemberDistance != candidates[j].BestMemberDistance {
			return candidates[i].BestMemberDistance < candidates[j].BestMemberDistance
		}
		return candidates[i].IdentityID < candidates[j].IdentityID
	})
}

func collectPotentialIdentityCandidates(item reviewSummaryItem, identities []PotentialIdentity) ([]potentialIdentityCandidate, []potentialIdentityCandidate) {
	if len(item.Vector) == 0 || len(identities) == 0 {
		return nil, nil
	}

	var eligible []potentialIdentityCandidate
	var overlapBlocked []potentialIdentityCandidate

	for idx, identity := range identities {
		candidate := potentialIdentityCandidateForItem(item, identity, idx)
		if candidate.OverlapBlockedTrackID != 0 {
			overlapBlocked = append(overlapBlocked, candidate)
			continue
		}
		if candidate.CentroidDistance <= potentialIdentityThreshold && candidate.BestMemberDistance <= potentialIdentityThreshold {
			eligible = append(eligible, candidate)
		}
	}

	sortPotentialIdentityCandidates(eligible)
	sortPotentialIdentityCandidates(overlapBlocked)
	return eligible, overlapBlocked
}

func buildPotentialIdentityLinkFromCandidates(eligible, overlapBlocked []potentialIdentityCandidate) PotentialIdentityLink {
	link := PotentialIdentityLink{Status: potentialIdentityStatusNew}
	if len(eligible) == 0 {
		if len(overlapBlocked) > 0 && overlapBlocked[0].Score <= potentialIdentityThreshold {
			blocked := overlapBlocked[0]
			link.OverlapBlockedPotentialIdentityID = blocked.IdentityID
			link.OverlapBlockedTrackID = blocked.OverlapBlockedTrackID
			link.OverlapBlockedDistance = roundTo(blocked.Score, 3)
		}
		return link
	}

	best := eligible[0]
	link.BestPotentialIdentityID = best.IdentityID
	link.BestPotentialIdentityScore = roundTo(best.Score, 3)
	link.BestMemberTrackID = best.BestMemberTrackID
	link.BestMemberDistance = roundTo(best.BestMemberDistance, 3)

	if len(eligible) > 1 {
		second := eligible[1]
		link.SecondBestPotentialIdentityID = second.IdentityID
		link.SecondBestPotentialIdentityScore = roundTo(second.Score, 3)
		if math.Abs(second.Score-best.Score) < potentialIdentityAmbiguityMargin {
			link.Status = potentialIdentityStatusAmbiguous
			return link
		}
	}

	if best.Score <= potentialIdentityStrongMatchThreshold {
		link.Status = potentialIdentityStatusStrong
	} else {
		link.Status = potentialIdentityStatusPossible
	}
	return link
}

func classifyPotentialIdentityLink(item reviewSummaryItem, identities []PotentialIdentity) PotentialIdentityLink {
	eligible, overlapBlocked := collectPotentialIdentityCandidates(item, identities)
	return buildPotentialIdentityLinkFromCandidates(eligible, overlapBlocked)
}

func findPotentialIdentityIndexByID(identities []PotentialIdentity, id int) int {
	for i := range identities {
		if identities[i].ID == id {
			return i
		}
	}
	return -1
}

func findPotentialIdentityMember(identities []PotentialIdentity, trackID int) (int, int) {
	for i := range identities {
		for j := range identities[i].Members {
			if identities[i].Members[j].Item.ID == trackID {
				return i, j
			}
		}
	}
	return -1, -1
}

func removeEmptyPotentialIdentities(identities []PotentialIdentity) []PotentialIdentity {
	filtered := identities[:0]
	for _, identity := range identities {
		if len(identity.Members) == 0 {
			continue
		}
		filtered = append(filtered, identity)
	}
	return filtered
}

func sortPotentialIdentityMembers(identities []PotentialIdentity) {
	for i := range identities {
		sort.Slice(identities[i].Members, func(a, b int) bool {
			if identities[i].Members[a].Item.StartTime != identities[i].Members[b].Item.StartTime {
				return identities[i].Members[a].Item.StartTime < identities[i].Members[b].Item.StartTime
			}
			return identities[i].Members[a].Item.ID < identities[i].Members[b].Item.ID
		})
	}
}

func sortPotentialIdentityUnresolved(unresolved []PotentialIdentityMember) {
	sort.Slice(unresolved, func(i, j int) bool {
		if unresolved[i].Item.StartTime != unresolved[j].Item.StartTime {
			return unresolved[i].Item.StartTime < unresolved[j].Item.StartTime
		}
		return unresolved[i].Item.ID < unresolved[j].Item.ID
	})
}

func clonePotentialIdentities(identities []PotentialIdentity) []PotentialIdentity {
	cloned := make([]PotentialIdentity, len(identities))
	for i, identity := range identities {
		cloned[i] = PotentialIdentity{
			ID:         identity.ID,
			Members:    append([]PotentialIdentityMember(nil), identity.Members...),
			SumVec:     cloneVector(identity.SumVec),
			Centroid:   cloneVector(identity.Centroid),
			TotalCount: identity.TotalCount,
		}
	}
	return cloned
}

func restorePotentialIdentityMember(identities []PotentialIdentity, identityID int, member PotentialIdentityMember) []PotentialIdentity {
	if idx := findPotentialIdentityIndexByID(identities, identityID); idx >= 0 {
		addPotentialIdentityMember(&identities[idx], member.Item, member.Link)
		return identities
	}
	identities = append(identities, newPotentialIdentityWithID(identityID, member.Item, member.Link))
	return identities
}

func findPotentialIdentityCandidateByID(candidates []potentialIdentityCandidate, identityID int) (potentialIdentityCandidate, bool) {
	for _, candidate := range candidates {
		if candidate.IdentityID == identityID {
			return candidate, true
		}
	}
	return potentialIdentityCandidate{}, false
}

func stagePossiblePotentialIdentityDecisions(identities []PotentialIdentity) []potentialIdentityDecision {
	possibleTrackIDs := collectPotentialIdentityMemberTrackIDsByStatus(identities, potentialIdentityStatusPossible)
	decisions := make([]potentialIdentityDecision, 0, len(possibleTrackIDs))

	for _, trackID := range possibleTrackIDs {
		snapshot := clonePotentialIdentities(identities)
		identityIdx, memberIdx := findPotentialIdentityMember(snapshot, trackID)
		if identityIdx < 0 || memberIdx < 0 {
			continue
		}

		originalIdentityID := snapshot[identityIdx].ID
		member := removePotentialIdentityMember(&snapshot[identityIdx], memberIdx)
		snapshot = removeEmptyPotentialIdentities(snapshot)

		eligible, overlapBlocked := collectPotentialIdentityCandidates(member.Item, snapshot)
		link := buildPotentialIdentityLinkFromCandidates(eligible, overlapBlocked)
		decision := potentialIdentityDecision{
			TrackID:            trackID,
			Kind:               potentialIdentityDecisionStay,
			OriginalIdentityID: originalIdentityID,
			Link:               member.Link,
		}

		if link.Status != potentialIdentityStatusStrong && link.Status != potentialIdentityStatusPossible {
			decision.Kind = potentialIdentityDecisionUnresolved
			decision.Link = link
			decisions = append(decisions, decision)
			continue
		}

		bestCandidate := eligible[0]
		if link.BestPotentialIdentityID == originalIdentityID {
			decision.Link = link
			decisions = append(decisions, decision)
			continue
		}

		currentCandidate, currentEligible := findPotentialIdentityCandidateByID(eligible, originalIdentityID)
		if currentEligible && bestCandidate.Score+potentialIdentityMoveMargin >= currentCandidate.Score {
			decision.Link = link
			decisions = append(decisions, decision)
			continue
		}

		decision.Kind = potentialIdentityDecisionMove
		decision.TargetIdentityID = link.BestPotentialIdentityID
		decision.Link = link
		decisions = append(decisions, decision)
	}

	sort.Slice(decisions, func(i, j int) bool {
		return decisions[i].TrackID < decisions[j].TrackID
	})
	return decisions
}

func stageUnresolvedPotentialIdentityDecisions(identities []PotentialIdentity, unresolved []PotentialIdentityMember) []potentialIdentityDecision {
	snapshot := clonePotentialIdentities(identities)
	sortedUnresolved := append([]PotentialIdentityMember(nil), unresolved...)
	sortPotentialIdentityUnresolved(sortedUnresolved)

	decisions := make([]potentialIdentityDecision, 0, len(sortedUnresolved))
	for _, member := range sortedUnresolved {
		link := classifyPotentialIdentityLink(member.Item, snapshot)
		decision := potentialIdentityDecision{
			TrackID: member.Item.ID,
			Kind:    potentialIdentityDecisionUnresolved,
			Link:    link,
		}
		if link.Status == potentialIdentityStatusStrong || link.Status == potentialIdentityStatusPossible {
			decision.Kind = potentialIdentityDecisionAttach
			decision.TargetIdentityID = link.BestPotentialIdentityID
		}
		decisions = append(decisions, decision)
	}
	return decisions
}

func applyPossiblePotentialIdentityDecisions(identities []PotentialIdentity, unresolved []PotentialIdentityMember, decisions []potentialIdentityDecision) ([]PotentialIdentity, []PotentialIdentityMember, bool) {
	changed := false

	for _, decision := range decisions {
		identityIdx, memberIdx := findPotentialIdentityMember(identities, decision.TrackID)
		if identityIdx < 0 || memberIdx < 0 {
			continue
		}

		switch decision.Kind {
		case potentialIdentityDecisionStay:
			identities[identityIdx].Members[memberIdx].Link = decision.Link
		case potentialIdentityDecisionMove:
			member := removePotentialIdentityMember(&identities[identityIdx], memberIdx)
			member.Link = decision.Link
			identities = restorePotentialIdentityMember(identities, decision.TargetIdentityID, member)
			changed = true
		case potentialIdentityDecisionUnresolved:
			member := removePotentialIdentityMember(&identities[identityIdx], memberIdx)
			member.Link = decision.Link
			unresolved = append(unresolved, member)
			changed = true
		}
	}

	identities = removeEmptyPotentialIdentities(identities)
	sortPotentialIdentityMembers(identities)
	sortPotentialIdentityUnresolved(unresolved)
	return identities, unresolved, changed
}

func applyUnresolvedPotentialIdentityDecisions(identities []PotentialIdentity, unresolved []PotentialIdentityMember, decisions []potentialIdentityDecision) ([]PotentialIdentity, []PotentialIdentityMember, bool) {
	if len(decisions) == 0 {
		sortPotentialIdentityMembers(identities)
		sortPotentialIdentityUnresolved(unresolved)
		return identities, unresolved, false
	}

	unresolvedByTrack := make(map[int]PotentialIdentityMember, len(unresolved))
	for _, member := range unresolved {
		unresolvedByTrack[member.Item.ID] = member
	}

	var remaining []PotentialIdentityMember
	changed := false

	for _, decision := range decisions {
		member, ok := unresolvedByTrack[decision.TrackID]
		if !ok {
			continue
		}

		switch decision.Kind {
		case potentialIdentityDecisionAttach:
			member.Link = decision.Link
			identities = restorePotentialIdentityMember(identities, decision.TargetIdentityID, member)
			changed = true
		default:
			member.Link = decision.Link
			remaining = append(remaining, member)
		}
	}

	identities = removeEmptyPotentialIdentities(identities)
	sortPotentialIdentityMembers(identities)
	sortPotentialIdentityUnresolved(remaining)
	return identities, remaining, changed
}

func potentialIdentityAssignmentSignature(identities []PotentialIdentity, unresolved []PotentialIdentityMember) string {
	var b strings.Builder
	for _, identity := range identities {
		b.WriteString(strconv.Itoa(identity.ID))
		b.WriteByte(':')
		for i, member := range identity.Members {
			if i > 0 {
				b.WriteByte(',')
			}
			b.WriteString(strconv.Itoa(member.Item.ID))
		}
		b.WriteByte(';')
	}
	b.WriteString("|u:")
	for i, member := range unresolved {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(strconv.Itoa(member.Item.ID))
	}
	return b.String()
}

func collectPotentialIdentityMemberTrackIDsByStatus(identities []PotentialIdentity, status PotentialIdentityStatus) []int {
	var trackIDs []int
	for _, identity := range identities {
		for _, member := range identity.Members {
			if member.Link.Status == status {
				trackIDs = append(trackIDs, member.Item.ID)
			}
		}
	}
	sort.Ints(trackIDs)
	return trackIDs
}

func refinePotentialIdentityAssignmentsUntilStable(identities []PotentialIdentity, unresolved []PotentialIdentityMember) ([]PotentialIdentity, []PotentialIdentityMember) {
	recomputeAllPotentialIdentityStatsFromMembers(identities)
	sortPotentialIdentityMembers(identities)
	sortPotentialIdentityUnresolved(unresolved)

	seenSignatures := make(map[string]struct{}, potentialIdentityRefinementMaxRounds+1)
	for round := 0; round < potentialIdentityRefinementMaxRounds; round++ {
		signature := potentialIdentityAssignmentSignature(identities, unresolved)
		if _, seen := seenSignatures[signature]; seen {
			break
		}
		seenSignatures[signature] = struct{}{}

		possibleDecisions := stagePossiblePotentialIdentityDecisions(identities)
		var possibleChanged bool
		identities, unresolved, possibleChanged = applyPossiblePotentialIdentityDecisions(identities, unresolved, possibleDecisions)
		recomputeAllPotentialIdentityStatsFromMembers(identities)
		sortPotentialIdentityMembers(identities)
		sortPotentialIdentityUnresolved(unresolved)

		unresolvedDecisions := stageUnresolvedPotentialIdentityDecisions(identities, unresolved)
		var unresolvedChanged bool
		identities, unresolved, unresolvedChanged = applyUnresolvedPotentialIdentityDecisions(identities, unresolved, unresolvedDecisions)
		recomputeAllPotentialIdentityStatsFromMembers(identities)
		sortPotentialIdentityMembers(identities)
		sortPotentialIdentityUnresolved(unresolved)

		if !possibleChanged && !unresolvedChanged {
			break
		}
	}

	recomputeAllPotentialIdentityStatsFromMembers(identities)
	sortPotentialIdentityMembers(identities)
	sortPotentialIdentityUnresolved(unresolved)
	return identities, unresolved
}

func buildPotentialIdentities(items []reviewSummaryItem) ([]PotentialIdentity, []PotentialIdentityMember) {
	if len(items) == 0 {
		return nil, nil
	}

	sortedItems := append([]reviewSummaryItem(nil), items...)
	sort.Slice(sortedItems, func(i, j int) bool {
		if sortedItems[i].StartTime != sortedItems[j].StartTime {
			return sortedItems[i].StartTime < sortedItems[j].StartTime
		}
		return sortedItems[i].ID < sortedItems[j].ID
	})

	var identities []PotentialIdentity
	var unresolved []PotentialIdentityMember
	nextPotentialIdentityID := 1

	for _, item := range sortedItems {
		link := classifyPotentialIdentityLink(item, identities)
		if link.Status == potentialIdentityStatusStrong || link.Status == potentialIdentityStatusPossible {
			if identityIndex := findPotentialIdentityIndexByID(identities, link.BestPotentialIdentityID); identityIndex >= 0 {
				addPotentialIdentityMember(&identities[identityIndex], item, link)
				continue
			}
		}

		if link.Status == potentialIdentityStatusAmbiguous {
			unresolved = append(unresolved, PotentialIdentityMember{
				Item: item,
				Link: link,
			})
			continue
		}

		identities = append(identities, newPotentialIdentity(nextPotentialIdentityID, item, link))
		nextPotentialIdentityID++
	}

	sortPotentialIdentityMembers(identities)
	identities, unresolved = refinePotentialIdentityAssignmentsUntilStable(identities, unresolved)

	return identities, unresolved
}

func potentialIdentityTrackList(members []PotentialIdentityMember) string {
	ids := make([]string, 0, len(members))
	for _, member := range members {
		ids = append(ids, strconv.Itoa(member.Item.ID))
	}
	return strings.Join(ids, ", ")
}

func potentialIdentityLinkLines(member PotentialIdentityMember) []string {
	switch member.Link.Status {
	case potentialIdentityStatusStrong, potentialIdentityStatusPossible:
		return []string{
			fmt.Sprintf("Track %d %s match to Potential Identity %d (best link: Track %d, distance: %.2f)", member.Item.ID, member.Link.Status, member.Link.BestPotentialIdentityID, member.Link.BestMemberTrackID, member.Link.BestMemberDistance),
		}
	case potentialIdentityStatusAmbiguous:
		return []string{
			fmt.Sprintf("Track %d AMBIGUOUS between:", member.Item.ID),
			fmt.Sprintf("Potential Identity %d (score: %.2f - BEST)", member.Link.BestPotentialIdentityID, member.Link.BestPotentialIdentityScore),
			fmt.Sprintf("Potential Identity %d (score: %.2f)", member.Link.SecondBestPotentialIdentityID, member.Link.SecondBestPotentialIdentityScore),
		}
	case potentialIdentityStatusNew:
		if member.Link.OverlapBlockedPotentialIdentityID > 0 {
			return []string{
				fmt.Sprintf("Track %d NEW potential identity (closest match to Potential Identity %d blocked by time overlap with Track %d, score: %.2f)", member.Item.ID, member.Link.OverlapBlockedPotentialIdentityID, member.Link.OverlapBlockedTrackID, member.Link.OverlapBlockedDistance),
			}
		}
	}
	return nil
}

func printPotentialIdentityLinkLines(lines []string) {
	if len(lines) == 0 {
		return
	}
	fmt.Fprintf(os.Stderr, "     - %s\n", lines[0])
	for _, line := range lines[1:] {
		fmt.Fprintf(os.Stderr, "       %s\n", line)
	}
}

func printReviewSummary(videoID string, items []reviewSummaryItem, totalDetections int) {
	identities, unresolved := buildPotentialIdentities(items)

	fmt.Fprintf(os.Stderr, "\n---------------------------------------------------------\n")
	fmt.Fprintf(os.Stderr, "📋 REVIEW SUMMARY [%s]\n", shortDisplayID(videoID))
	fmt.Fprintf(os.Stderr, "---------------------------------------------------------\n")

	for _, identity := range identities {
		fmt.Fprintf(os.Stderr, "\n👤 Potential Identity %d\n", identity.ID)
		fmt.Fprintf(os.Stderr, "   tracks: %s\n", potentialIdentityTrackList(identity.Members))
		fmt.Fprintf(os.Stderr, "   spans:\n")
		for _, member := range identity.Members {
			fmt.Fprintf(os.Stderr, "     - Track %d: %s -> %s [action: %s]\n", member.Item.ID, utils.FmtTime(member.Item.StartTime), utils.FmtTime(member.Item.EndTime), reviewActionLabel(member.Item.Action))
		}

		var linkageMembers []PotentialIdentityMember
		for _, member := range identity.Members {
			if len(potentialIdentityLinkLines(member)) > 0 {
				linkageMembers = append(linkageMembers, member)
			}
		}
		if len(linkageMembers) > 0 {
			fmt.Fprintf(os.Stderr, "   linkage:\n")
			for _, member := range linkageMembers {
				printPotentialIdentityLinkLines(potentialIdentityLinkLines(member))
			}
		}
	}

	for _, member := range unresolved {
		fmt.Fprintf(os.Stderr, "\n👤 Unresolved Track %d\n", member.Item.ID)
		fmt.Fprintf(os.Stderr, "   time: %s -> %s\n", utils.FmtTime(member.Item.StartTime), utils.FmtTime(member.Item.EndTime))
		fmt.Fprintf(os.Stderr, "   action: %s\n", reviewActionLabel(member.Item.Action))
		fmt.Fprintf(os.Stderr, "   linkage:\n")
		printPotentialIdentityLinkLines(potentialIdentityLinkLines(member))
	}

	fmt.Fprintf(os.Stderr, "\n---------------------------------------------------------\n")
	fmt.Fprintf(os.Stderr, "👁️  Total Face Detections:   %d\n", totalDetections)
	fmt.Fprintf(os.Stderr, "---------------------------------------------------------\n")
}

func processResults(ctx context.Context, results <-chan scanResult, db scanDB, videoID, reviewID string, fps float64, opts Options, errChan chan<- error, finalIntervalsChan chan<- []store.IntervalData, inflightSem chan struct{}, startupReady chan<- struct{}) {
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
				SummaryItems: reviewSummaryItems,
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
	outputBase := currentOutputBase()
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
	fmt.Fprintf(os.Stderr, "📂 Output Directory [%s]: %s\n", shortDisplayID(videoID), userFacingOutputPath(resultsDir))

	fmt.Fprintf(os.Stderr, "⚙️  Tracker initialized with Cosine Distance (Threshold: %.2f)\n", opts.MatchThreshold)
	close(startupReady)

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
				filename:   headlineArtifactFilename(1, "First_Detection", t.StartFrame, t.FirstScore),
				data:       t.FirstThumb,
				removeGlob: "1_First_Detection_*.jpg",
			}) {
				return
			}
			if !sendThumbOp(thumbOp{
				dir:        trackDir,
				filename:   headlineArtifactFilename(2, "Last_Detection", t.LastFrame, t.LastScore),
				data:       t.LastThumb,
				removeGlob: "2_Last_Detection_*.jpg",
			}) {
				return
			}
			if !sendThumbOp(thumbOp{
				dir:        trackDir,
				filename:   headlineArtifactFilename(3, "Highest_Confidence", t.BestFrame, t.BestQuality),
				data:       t.BestThumb,
				removeGlob: "3_Highest_Confidence_*.jpg",
			}) {
				return
			}
			if !sendThumbOp(thumbOp{
				dir:        trackDir,
				filename:   headlineArtifactFilename(4, "Lowest_Confidence", t.LowestFrame, t.LowestScore),
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
			reviewItems = append(reviewItems, item)
			reviewSummaryItems = append(reviewSummaryItems, reviewSummaryItem{
				ID:                trackReviewID,
				StartTime:         startSec,
				EndTime:           endSec,
				NearestCandidates: item.NearestCandidates,
				Identity:          item.Identity,
				Variant:           item.Variant,
				Confidence:        item.Confidence,
				Action:            item.Action,
				ArtifactDir:       reviewArtifactRelativeDir(videoID, reviewID, trackReviewID),
				Vector:            cloneVector(t.MeanVec),
				Count:             t.Count,
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
				t.BestFrame = t.TopFrames[bestIdx].Index
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
				filename:   headlineArtifactFilename(1, "First_Detection", t.StartFrame, t.FirstScore),
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
			filename:   headlineArtifactFilename(2, "Last_Detection", t.LastFrame, t.LastScore),
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
				filename:   headlineArtifactFilename(3, "Highest_Confidence", t.BestFrame, t.BestQuality),
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
				filename:   headlineArtifactFilename(4, "Lowest_Confidence", t.LowestFrame, t.LowestScore),
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
			t.BestFrame = frame.Index
		}

		candidate := frameCandidate{Index: frame.Index, Score: face.Quality, Vec: face.Vec, Thumb: face.Thumb}
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
			t.LowestScore, t.LowestThumb, t.LowestFrame = face.Quality, face.Thumb, frame.Index
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

func splitReviewDocument(doc ReviewDocument) (ReviewFileDocument, ReviewSidecarDocument, error) {
	reviewDoc := buildReviewFileDocument(doc)
	reviewData := ReviewSidecarDocument{
		ReviewID:  reviewDocumentID(doc),
		VideoID:   doc.VideoID,
		InputPath: doc.InputPath,
		Tracks:    make(map[string]ReviewTrackData, len(doc.Tracks)),
	}

	for _, item := range doc.Tracks {
		if len(item.InternalVector) == 0 && item.InternalCount == 0 {
			continue
		}
		key := stagingItemKey(item)
		if key == "" {
			return ReviewFileDocument{}, ReviewSidecarDocument{}, fmt.Errorf("cannot write review data for track with blank id")
		}
		fingerprint, err := reviewTrackFingerprint(item)
		if err != nil {
			return ReviewFileDocument{}, ReviewSidecarDocument{}, err
		}
		reviewData.Tracks[key] = ReviewTrackData{
			Fingerprint:    fingerprint,
			InternalVector: item.InternalVector,
			InternalCount:  item.InternalCount,
		}
	}

	return reviewDoc, reviewData, nil
}

func writeReviewFile(path string, doc ReviewFileDocument) error {
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
	header := "# Leave top-level `raw_tracks`, `unresolved_tracks`, `review_id`, `video_id`, and `input_path` unchanged.\n" +
		"# Edit only `potential_identities[].tracks`, `potential_identities[].identity`, `potential_identities[].variant`, and `potential_identities[].action`.\n" +
		"# `review_id` is the unique identifier for this scan run and review file.\n" +
		"# Each `potential_identities[].tracks` entry must use either a <track_id> or an inclusive <start>-<end> range of track IDs listed under `raw_tracks`.\n" +
		"# `unresolved_tracks` is read-only. To resolve one, add its <track_id> to a `potential_identities[].tracks` entry or create a new potential identity entry.\n" +
		"# Leave `raw_tracks.*.start_time`, `raw_tracks.*.end_time`, `raw_tracks.*.nearest_candidates`, `raw_tracks.*.confidence`, `raw_tracks.*.suggested_identity`, `raw_tracks.*.suggested_variant`, `raw_tracks.*.suggested_action`, and `raw_tracks.*.artifact_dir` unchanged.\n" +
		"# Leave each `potential_identities[].id` unchanged.\n" +
		"# Valid actions: merge | new_identity | new_variant | discard.\n" +
		"# The prefilled `action` is Sentinel's best guess from scan heuristics, not final truth.\n"
	if doc.ArtifactRoot != "" {
		header += "# Each `raw_tracks.<track_id>.artifact_dir` points to corresponding thumbnails and frame artifacts under:\n"
		header += fmt.Sprintf("#   %s/<track_id>/\n", doc.ArtifactRoot)
	}
	formatted = header + formatted
	formatted = strings.ReplaceAll(formatted, "\nraw_tracks:\n", "\nraw_tracks:\n\n    # Read-only track evidence below is generated by scan.\n")
	formatted = strings.ReplaceAll(formatted, "\nunresolved_tracks:\n", "\nunresolved_tracks:\n\n    # Read-only unresolved track IDs below were not confidently grouped during scan.\n")
	formatted = strings.ReplaceAll(formatted, "\npotential_identities:\n", "\npotential_identities:\n\n    # Editable potential identities below control track membership and the values that will be committed.\n")
	formatted = strings.ReplaceAll(formatted, "\n    - id:", "\n\n    # Leave `id` unchanged.\n    - id:")
	formatted = strings.ReplaceAll(formatted, "\n      tracks:", "\n      # Edit `tracks` to move <track_id> entries or <start>-<end> ranges between potential identities.\n      # For `new_identity`, if `identity` is left blank, then Sentinel will auto-name it. If `variant` is left blank, then Sentinel will create the `Default` variant.\n      # For `new_variant`, set `identity` to the existing person and `variant` to the new variant name.\n      tracks:")
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
		BestFrame:      frameIndex,
		FirstThumb:     thumb,
		FirstScore:     quality,
		LastThumb:      thumb,
		LastScore:      quality,
		LowestThumb:    thumb,
		LowestScore:    quality,
		LowestFrame:    frameIndex,
		LastSavedScore: quality,
		PendingFrames:  []frameData{{Index: frameIndex, Score: quality, Data: thumb}},
		TopFrames:      []frameCandidate{{Index: frameIndex, Score: quality, Vec: vec, Thumb: thumb}},
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
	outputBase := currentOutputBase()

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
