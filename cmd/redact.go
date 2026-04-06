package cmd

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"io"
	"math"
	"os"
	"path/filepath"
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
)

var (
	redactOpts           Options
	redactOutput         string
	redactMode           string
	redactTargets        string
	redactStyle          string
	redactLinger         string
	redactParanoid       bool
	redactParanoidStrict bool
	redactBufferSize     int
)

const defaultRedactionBoxScale = 1.2

var redactCmd = &cobra.Command{
	Use:   "redact",
	Short: "Redact all faces or specific identity IDs from a video",
	RunE: func(cmd *cobra.Command, args []string) error {
		cmd.SilenceUsage = true
		return runRedact(cmd.Context(), cmd, redactOpts)
	},
}

func init() {
	redactCmd.Flags().StringVarP(&redactOpts.InputPath, "input", "i", "", "Path to input video")
	redactCmd.Flags().StringVarP(&redactOutput, "output", "o", "output/redacted.mp4", "Path to output video")
	redactCmd.Flags().StringVarP(&redactMode, "mode", "m", "blur-all", "Redaction mode: blur-all, targeted")
	redactCmd.Flags().StringVar(&redactTargets, "target", "", "Comma-separated list of Identity IDs to redact (implies targeted mode)")
	redactCmd.Flags().StringVar(&redactStyle, "style", "black", "Redaction style: pixel, black, gauss, secure")
	redactCmd.Flags().StringVar(&redactLinger, "linger", "1s", "How long to keep blurring after a targeted face is lost")
	redactCmd.Flags().BoolVar(&redactParanoid, "paranoid", false, "Enable paranoid mode: blur ALL faces if a targeted face is lost")
	redactCmd.Flags().BoolVar(&redactParanoidStrict, "paranoid-strict", false, "In paranoid mode, trigger blurring even if a target has not appeared yet")

	redactCmd.Flags().IntVarP(&redactOpts.NumEngines, "engines", "e", 1, "Number of parallel engine workers")
	redactCmd.Flags().Float64VarP(&redactOpts.MatchThreshold, "threshold", "t", 0.6, "Face matching threshold")
	redactCmd.Flags().Float64VarP(&redactOpts.DetectionThreshold, "detection-threshold", "D", 0.5, "Face detection confidence threshold")
	redactCmd.Flags().IntVarP(&redactOpts.BlurStrength, "strength", "s", 15, "Pixelation block size (higher = more blocky)")
	redactCmd.Flags().Float64Var(&redactOpts.BoxScale, "box-scale", defaultRedactionBoxScale, "Scale factor for redaction boxes (1.0 = detected box size)")
	redactCmd.Flags().StringVar(&redactOpts.WorkerTimeout, "worker-timeout", "30s", "Timeout for a worker to process a single frame")
	redactCmd.Flags().IntVarP(&redactBufferSize, "buffer-size", "B", 35, "Max number of frames to buffer in memory (prevents OOM)")

	redactCmd.MarkFlagRequired("input")
	rootCmd.AddCommand(redactCmd)
}

// blurBufferPool recycles scratch buffers for the Gaussian blur.
var blurBufferPool = sync.Pool{
	New: func() interface{} { return make([]uint8, 0, 1024*1024) }, // Start with 1MB capacity
}

type redactResult struct {
	Index int
	Data  []byte
	Faces []types.FaceResult
}

// redactionTrack holds the state for a single targeted identity during redaction.
type redactionTrack struct {
	ID        int
	LastSeen  int // Frame index
	LastKnown image.Rectangle
}

// paranoidTracker manages the stateful redaction logic, including lingering and paranoid mode.
type paranoidTracker struct {
	targetVariants []store.VariantData // All variants for target masters
	activeTracks   []*redactionTrack
	opts           *Options
}

func runRedact(ctx context.Context, cmd *cobra.Command, opts Options) error {
	// Create a cancellable context to ensure all child processes (FFmpeg, Python)
	// are killed immediately if this function returns early (e.g. on error).
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	mode := effectiveRedactMode(cmd)

	if err := validateRedactFlags(cmd, &opts); err != nil {
		return err
	}

	// Ensure paranoid mode is active if strict mode is requested
	activeParanoid := redactParanoid
	if redactParanoidStrict {
		activeParanoid = true
	}

	// Safety Check: Prevent overwriting input file which causes corruption
	inAbs, err := filepath.Abs(opts.InputPath)
	if err != nil {
		return fmt.Errorf("failed to resolve absolute path for input: %w", err)
	}
	outAbs, err := filepath.Abs(redactOutput)
	if err != nil {
		return fmt.Errorf("failed to resolve absolute path for output: %w", err)
	}
	if inAbs == outAbs {
		return fmt.Errorf("input and output paths must be different to prevent file corruption")
	}

	targetIDs := []int{}
	var validationErrors []string
	if mode == "targeted" {
		for _, s := range strings.Split(redactTargets, ",") {
			id, err := strconv.Atoi(strings.TrimSpace(s))
			if err != nil {
				validationErrors = append(validationErrors, fmt.Sprintf("'%s'", s))
				continue
			}
			targetIDs = append(targetIDs, id)
		}
	}
	if len(validationErrors) > 0 {
		return fmt.Errorf("invalid target IDs: %s", strings.Join(validationErrors, ", "))
	}

	if err := os.MkdirAll(filepath.Dir(redactOutput), 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}

	fps, err := utils.GetVideoFPS(ctx, opts.InputPath)
	if err != nil {
		return fmt.Errorf("failed to determine video FPS: %w", err)
	}
	if fps <= 0 || math.IsNaN(fps) || math.IsInf(fps, 0) {
		return fmt.Errorf("invalid video FPS: %f (must be > 0)", fps)
	}
	width, height, err := utils.GetVideoDimensions(ctx, opts.InputPath)
	if err != nil {
		return fmt.Errorf("failed to determine video dimensions: %w", err)
	}
	totalFrames := utils.GetTotalFrames(ctx, opts.InputPath)

	taskChan := make(chan types.FrameTask, opts.NumEngines)
	resultsChan := make(chan redactResult, opts.NumEngines*2)
	writeChan := make(chan []byte, opts.NumEngines*2)
	writeErrChan := make(chan error, 1)
	errChan := make(chan error, opts.NumEngines+2)

	var wg sync.WaitGroup
	var writerWg sync.WaitGroup
	var bufferedFrames int64
	readyChan := make(chan struct{}, opts.NumEngines)
	releaseBufferedFrame := func() {
		atomic.AddInt64(&bufferedFrames, -1)
	}
	putBufferedFrame := func(buf []byte) {
		frameBufferPool.Put(buf)
		releaseBufferedFrame()
	}

	// Safety Net: Cleanup Deferred Block to prevent Memory Leaks (Dangling Rows) & Deadlocks
	defer func() {
		cancel()        // 1. Signal shutdown
		wg.Wait()       // 2. Wait for workers
		writerWg.Wait() // 3. Wait for writer

		// 4. Drain Channels (Return buffers to pool)
	DrainTask:
		for {
			select {
			case t, ok := <-taskChan:
				if !ok {
					break DrainTask
				}
				putBufferedFrame(t.Data)
			default:
				break DrainTask
			}
		}
	DrainResults:
		for {
			select {
			case r, ok := <-resultsChan:
				if !ok {
					break DrainResults
				}
				putBufferedFrame(r.Data)
			default:
				break DrainResults
			}
		}
	DrainWrite:
		for {
			select {
			case d, ok := <-writeChan:
				if !ok {
					break DrainWrite
				}
				putBufferedFrame(d)
			default:
				break DrainWrite
			}
		}
	}()

	// Optimization: Load target variant embeddings only when targeted matching is needed.
	targetVariants, err := loadRedactTargetVariants(ctx, mode, targetIDs)
	if err != nil {
		return fmt.Errorf("failed to load target embeddings: %w", err)
	}

	// Critical Safety Check: Ensure ALL requested targets actually exist in the DB.
	// If a target is missing, we must abort to prevent accidental leakage (unredacted faces).
	if mode == "targeted" {
		foundIdentityIDs := make(map[int]bool)
		for _, v := range targetVariants {
			foundIdentityIDs[v.IdentityID] = true
		}
		for _, id := range targetIDs {
			if !foundIdentityIDs[id] {
				return fmt.Errorf("aborting: target Identity ID %d not found in database", id)
			}
		}
	}

	lingerDuration, err := time.ParseDuration(redactLinger)
	if err != nil {
		return fmt.Errorf("failed to parse linger duration: %w", err)
	}
	lingerFrames := int(lingerDuration.Seconds() * fps)
	tracker := &paranoidTracker{
		targetVariants: targetVariants,
		activeTracks:   make([]*redactionTrack, 0),
		opts:           &opts,
	}

	workerTimeout, err := time.ParseDuration(opts.WorkerTimeout)
	if err != nil {
		return fmt.Errorf("failed to parse worker timeout: %w", err)
	}

	// Flow Control: Prevent OOM by limiting in-flight frames.
	// This is a CAP (Semaphore), not an upfront allocation. However, since decoding is faster
	// than inference, the buffer will typically fill up to this limit during operation.
	bufferSize := redactBufferSize
	if !cmd.Flags().Changed("buffer-size") {

		totalPixels := width * height

		if totalPixels <= 2048*1080 { //1080p
			bufferSize = 200
			fmt.Fprintf(os.Stderr, "⚡ Standard-Res Video detected. Auto-setting buffer to %d frames (Max ~1.5GB) for performance.\n", bufferSize)

		} else if totalPixels <= 4096*2160 { // 4K (Cover DCI 4K)
			bufferSize = 45
			fmt.Fprintf(os.Stderr, "⚠️  4K High-Res Video detected. Auto-setting buffer to %d frames (Max ~1.5GB) to prevent OOM.\n", bufferSize)

		} else if totalPixels <= 8192*4320 { // 8K (Cover DCI 8K)
			bufferSize = 11
			fmt.Fprintf(os.Stderr, "⚠️  8K High-Res Video detected. Auto-setting buffer to %d frames (Max ~1.5GB) to prevent OOM.\n", bufferSize)

		} else { // 8K+
			bufferSize = 3
			fmt.Fprintf(os.Stderr, "⚠️  Extreme-Res Video detected. Auto-setting buffer to %d frames (Max ~1.5GB for 16K) to prevent OOM.\n", bufferSize)

		}
	}

	if bufferSize < 1 {
		bufferSize = 1
	}
	inflightSem := make(chan struct{}, bufferSize)
	var currentMemory uint64
	pidChan := make(chan int, opts.NumEngines)

	go func() {
		var pids []int
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
			total += utils.GetProcessRSS(os.Getpid())
			for _, pid := range pids {
				total += utils.GetProcessRSS(pid)
			}
			atomic.StoreUint64(&currentMemory, total)
		}

		updateMem()
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

	for i := 0; i < opts.NumEngines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()

			inferenceMode := "full"
			if mode == "blur-all" {
				inferenceMode = "detection-only"
			}

			cfg := worker.RedactConfig{
				DetectionThreshold: opts.DetectionThreshold,
				ReadTimeout:        workerTimeout,
				InferenceMode:      inferenceMode,
				RawWidth:           width,
				RawHeight:          height,
			}
			pyWorker, err := worker.NewPythonRedactWorker(ctx, id, cfg)
			if err != nil {
				select {
				case <-ctx.Done():
					return
				case errChan <- &utils.ContextualError{Context: "Worker Startup Failed", Err: err}:
					return
				}
			}
			defer pyWorker.Close()
			select {
			case pidChan <- pyWorker.Cmd.Process.Pid:
			case <-ctx.Done():
				return
			}
			select {
			case readyChan <- struct{}{}:
			case <-ctx.Done():
				return
			}

			for {
				select {
				case <-ctx.Done():
					return
				case task, ok := <-taskChan:
					if !ok {
						return
					}
					faces, err := pyWorker.ProcessRedactFrame(task.Data)
			if err != nil {
				putBufferedFrame(task.Data) // Return buffer on error
				pyWorker.Close()               // Reap process before diagnostics
						utils.ShowError("Python crashed", err, pyWorker.Cmd)
						select {
						case <-ctx.Done():
							return
						case errChan <- &utils.SilentError{Err: err}:
							return
						}
					}
					select {
					case resultsChan <- redactResult{Index: task.Index, Data: task.Data, Faces: faces}:
					case <-ctx.Done():
						putBufferedFrame(task.Data) // Return buffer on cancellation
						return
					}
				}
			}
		}(i)
	}

	fmt.Fprintln(os.Stderr, "🚀 Warming up engines...")
	for i := 0; i < opts.NumEngines; i++ {
		select {
		case <-readyChan:
		case err := <-errChan:
			return err
		case <-ctx.Done():
			return ctx.Err()
		}
	}

	var decoderStderr bytes.Buffer
	decoder := utils.NewFFmpegRawDecoder(ctx, opts.InputPath)
	decoder.Stderr = &decoderStderr
	decoderOut, err := decoder.StdoutPipe()
	if err != nil {
		return fmt.Errorf("failed to create decoder pipe: %w", err)
	}
	defer decoderOut.Close()
	if err := decoder.Start(); err != nil {
		return fmt.Errorf("failed to start decoder: %w", err)
	}
	// Ensure we reap the process even if we return early
	decoderWaited := false
	defer func() {
		if !decoderWaited {
			cancel() // Kill process to prevent deadlock on Wait()
			decoder.Wait()
		}
	}()

	var encoderStderr bytes.Buffer
	// Use the utility which now handles audio mapping correctly
	encoder := utils.NewFFmpegEncoder(ctx, opts.InputPath, redactOutput, fps, width, height)
	encoder.Stderr = &encoderStderr
	encoderIn, err := encoder.StdinPipe()
	if err != nil {
		return fmt.Errorf("failed to create encoder pipe: %w", err)
	}
	defer encoderIn.Close()
	if err := encoder.Start(); err != nil {
		return fmt.Errorf("failed to start encoder: %w", err)
	}
	// Ensure we reap the process even if we return early
	encoderWaited := false
	defer func() {
		if !encoderWaited {
			cancel() // Kill process to prevent deadlock on Wait()
			encoder.Wait()
		}
	}()

	go func() {
		frameSize := width * height * 4
		idx := 0
		defer close(taskChan) // Fix: Ensure channel is closed even on early return
		for {
			// We use the shared pool from scan.go to reduce GC pressure
			buf := frameBufferPool.Get().([]byte)
			if cap(buf) < frameSize {
				buf = make([]byte, frameSize)
			}
			buf = buf[:frameSize]

			_, err := io.ReadFull(decoderOut, buf)
			if err != nil {
				// EOF or unexpected error, stop reading
				frameBufferPool.Put(buf)
				break
			}
			atomic.AddInt64(&bufferedFrames, 1)

			// Acquire semaphore (Block if too many frames are in flight)
			select {
			case inflightSem <- struct{}{}:
			case <-ctx.Done():
				putBufferedFrame(buf)
				return
			}

			select {
			case taskChan <- types.FrameTask{Index: idx, Data: buf}:
				idx++
			case <-ctx.Done():
				// We acquired the semaphore, but are exiting before the frame is processed.
				// Release the semaphore and the buffer to prevent leaks.
				<-inflightSem
				putBufferedFrame(buf)
				return
			}
		}
	}()

	buffer := make(map[int]redactResult)
	// Ensure buffered frames are returned to the pool if we exit early on error
	defer func() {
		for _, res := range buffer {
			putBufferedFrame(res.Data)
		}
	}()
	nextFrame := 0
	var barTotal int64 = int64(totalFrames)
	if barTotal <= 0 {
		barTotal = -1 // Trigger spinner mode
	}
	bar := progressbar.NewOptions64(barTotal,
		progressbar.OptionSetDescription("Redacting"),
		progressbar.OptionSetWriter(os.Stderr),
		progressbar.OptionShowCount(),
	)
	barUpdateStop := make(chan struct{})
	barUpdateDone := make(chan struct{})
	go func() {
		defer close(barUpdateDone)
			updateDesc := func() {
				mem := atomic.LoadUint64(&currentMemory)
				inFlight := len(inflightSem)
				inFlightCap := cap(inflightSem)
				buffered := atomic.LoadInt64(&bufferedFrames)
				if mem > 0 {
					bar.Describe(fmt.Sprintf("Redacting (RAM: %.2f GB | Buffered: %d | In Flight: %d/%d)", float64(mem)/(1024*1024*1024), buffered, inFlight, inFlightCap))
					return
				}
				bar.Describe(fmt.Sprintf("Redacting (Buffered: %d | In Flight: %d/%d)", buffered, inFlight, inFlightCap))
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
	defer func() {
		close(barUpdateStop)
		<-barUpdateDone
		_ = bar.Finish()
		fmt.Fprintln(os.Stderr)
	}()

	go func() {
		wg.Wait()
		close(resultsChan)
	}()

	// Decoupled writer to prevent deadlock
	writerWg.Add(1)
	go func() {
		defer writerWg.Done()
		var writeError error
		for {
			select {
			case data, ok := <-writeChan:
				if !ok {
					return
				}
				if writeError == nil {
					if _, err := encoderIn.Write(data); err != nil {
						writeError = err
						select {
						case <-ctx.Done():
							return
						case writeErrChan <- err:
							return
						}
					}
				}
				putBufferedFrame(data)
			case <-ctx.Done():
				return
			}
		}
	}()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case err := <-errChan:
			return err
		case err := <-writeErrChan:
			return err
		case res, ok := <-resultsChan:
			if !ok {
				goto Flush
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

				outData, err := tracker.apply(ctx, frame.Data, width, height, nextFrame, frame.Faces, mode, redactStyle, activeParanoid, redactParanoidStrict, lingerFrames)
				if err != nil {
					putBufferedFrame(frame.Data)
					return fmt.Errorf("redaction failed: %w", err)
				}

				select {
				case writeChan <- outData:
				case err := <-writeErrChan:
					putBufferedFrame(outData)
					return err
				case <-ctx.Done():
					// The frame was processed but not sent to the encoder. Recycle the buffer before exiting.
					putBufferedFrame(outData)
					return ctx.Err()
				}

				// Release semaphore, allowing decoder to read more frames
				<-inflightSem

				bar.Add(1)
				nextFrame++
			}
		}
	}

Flush:
	if len(buffer) > 0 {
		return fmt.Errorf("redaction stopped with missing frame %d; %d later frame(s) remained buffered", nextFrame, len(buffer))
	}

	close(writeChan)
	writerWg.Wait()

	// Check for any errors that occurred in the writer during the final flush
	select {
	case err := <-writeErrChan:
		return err
	default:
	}

	if err := encoderIn.Close(); err != nil {
		return fmt.Errorf("failed to close encoder pipe: %w", err)
	}
	encoderWaited = true
	if err := encoder.Wait(); err != nil {
		if encoderStderr.Len() > 0 {
			fmt.Fprintf(os.Stderr, "\nFFmpeg Encoder Logs:\n%s\n", encoderStderr.String())
		}
		return fmt.Errorf("encoder process failed: %w", err)
	}
	decoderWaited = true
	if err := decoder.Wait(); err != nil {
		if decoderStderr.Len() > 0 {
			fmt.Fprintf(os.Stderr, "\nFFmpeg Decoder Logs:\n%s\n", decoderStderr.String())
		}
		return fmt.Errorf("decoder process failed: %w", err)
	}
	return nil
}

func (p *paranoidTracker) apply(ctx context.Context, imgData []byte, width, height, frameIndex int, faces []types.FaceResult, mode, style string, paranoid bool, paranoidStrict bool, lingerFrames int) ([]byte, error) {
	// Zero-Copy: Wrap the raw bytes in an image.RGBA struct
	m := &image.RGBA{
		Pix:    imgData,
		Stride: width * 4,
		Rect:   image.Rect(0, 0, width, height),
	}
	strength := p.opts.BlurStrength
	boxScale := p.effectiveBoxScale()
	bounds := m.Bounds()

	// Map to track which face index corresponds to which target ID (for blurring later)
	faceIdentities := make([]int, len(faces))
	for i := range faceIdentities {
		faceIdentities[i] = -1
	}
	matchedRectsByIdentity := make(map[int][]image.Rectangle)

	for i, face := range faces {
		if mode == "targeted" {
			bestID := -1
			minDist := p.opts.MatchThreshold

			for _, v := range p.targetVariants {
				dist := utils.CosineDist(face.Vec, v.Vec)
				if dist < minDist {
					minDist = dist
					bestID = v.IdentityID
				}
			}

			if bestID != -1 {
				rect := image.Rect(face.Loc[0], face.Loc[1], face.Loc[2], face.Loc[3])
				faceIdentities[i] = bestID
				matchedRectsByIdentity[bestID] = append(matchedRectsByIdentity[bestID], rect)
			}
		}
	}
	if mode == "targeted" {
		p.updateTracks(frameIndex, matchedRectsByIdentity)
	}

	isParanoidActive := false
	if paranoid && mode == "targeted" {
		if paranoidStrict {
			// Strict mode: trigger if any target identity is missing, period.
			isParanoidActive = p.anyTargetIdentityMissing(matchedRectsByIdentity)
		} else {
			// Default behavior: trigger if any tracked target instance is missing.
			isParanoidActive = p.hasMissingTrackedInstance(frameIndex, lingerFrames)
		}
	}

	for i, face := range faces {
		shouldBlur := false
		if mode == "blur-all" || isParanoidActive {
			shouldBlur = true
		} else if mode == "targeted" {
			if faceIdentities[i] != -1 {
				shouldBlur = true
			}
		}

		if shouldBlur {
			rect := scaledRedactionRect(image.Rect(face.Loc[0], face.Loc[1], face.Loc[2], face.Loc[3]), boxScale, bounds)
			redactFace(m, rect, style, strength)
		}
	}

	// Apply Linger (only in targeted mode)
	// If paranoid is active, we are blurring all detected faces.
	// However, linger covers *undetected* faces (lost tracking), so we still apply it.
	if mode == "targeted" {
		for _, track := range p.activeTracks {
			if track.LastSeen != frameIndex && frameIndex-track.LastSeen <= lingerFrames {
				redactFace(m, scaledRedactionRect(track.LastKnown, boxScale, bounds), style, strength)
			}
		}
	}

	p.pruneTracks(frameIndex, lingerFrames)

	return m.Pix, nil
}

func (p *paranoidTracker) effectiveBoxScale() float64 {
	if p == nil || p.opts == nil || p.opts.BoxScale <= 0 {
		return defaultRedactionBoxScale
	}
	return p.opts.BoxScale
}

func loadRedactTargetVariants(ctx context.Context, mode string, targetIDs []int) ([]store.VariantData, error) {
	if mode != "targeted" {
		return nil, nil
	}
	if DB == nil {
		return nil, fmt.Errorf("targeted mode requires database connection")
	}
	return DB.GetVariantsForIdentities(ctx, targetIDs)
}

func effectiveRedactMode(cmd *cobra.Command) string {
	mode := redactMode
	targets := strings.TrimSpace(redactTargets)
	modeExplicit := false

	if cmd != nil {
		if flag := cmd.Flags().Lookup("mode"); flag != nil {
			modeExplicit = cmd.Flags().Changed("mode")
		}
		if resolvedMode, err := cmd.Flags().GetString("mode"); err == nil && resolvedMode != "" {
			mode = resolvedMode
		}
		if resolvedTargets, err := cmd.Flags().GetString("target"); err == nil {
			targets = strings.TrimSpace(resolvedTargets)
		}
	}

	if targets != "" && !modeExplicit && (mode == "" || mode == "blur-all") {
		return "targeted"
	}
	if mode == "" {
		return "blur-all"
	}
	return mode
}

func (p *paranoidTracker) updateTracks(frameIndex int, matchedRectsByIdentity map[int][]image.Rectangle) {
	for identityID, rects := range matchedRectsByIdentity {
		candidates := p.activeTracksForIdentity(identityID)
		used := make(map[*redactionTrack]bool)

		for _, rect := range rects {
			track := closestRedactionTrack(candidates, used, rect)
			if track == nil {
				track = &redactionTrack{ID: identityID}
				p.activeTracks = append(p.activeTracks, track)
			}
			track.LastSeen = frameIndex
			track.LastKnown = rect
			used[track] = true
		}
	}
}

func (p *paranoidTracker) activeTracksForIdentity(identityID int) []*redactionTrack {
	tracks := make([]*redactionTrack, 0)
	for _, track := range p.activeTracks {
		if track.ID == identityID {
			tracks = append(tracks, track)
		}
	}
	return tracks
}

func (p *paranoidTracker) anyTargetIdentityMissing(matchedRectsByIdentity map[int][]image.Rectangle) bool {
	for _, variant := range p.targetVariants {
		if len(matchedRectsByIdentity[variant.IdentityID]) == 0 {
			return true
		}
	}
	return false
}

func (p *paranoidTracker) hasMissingTrackedInstance(frameIndex, lingerFrames int) bool {
	for _, track := range p.activeTracks {
		if track.LastSeen != frameIndex && frameIndex-track.LastSeen <= lingerFrames {
			return true
		}
	}
	return false
}

func closestRedactionTrack(candidates []*redactionTrack, used map[*redactionTrack]bool, rect image.Rectangle) *redactionTrack {
	var best *redactionTrack
	var bestDistance int64

	for _, candidate := range candidates {
		if used[candidate] {
			continue
		}

		distance := rectCenterDistanceSquared(candidate.LastKnown, rect)
		if best == nil || distance < bestDistance {
			best = candidate
			bestDistance = distance
		}
	}

	return best
}

func rectCenterDistanceSquared(a, b image.Rectangle) int64 {
	ax := int64(a.Min.X + a.Max.X)
	ay := int64(a.Min.Y + a.Max.Y)
	bx := int64(b.Min.X + b.Max.X)
	by := int64(b.Min.Y + b.Max.Y)

	dx := ax - bx
	dy := ay - by
	return dx*dx + dy*dy
}

func scaledRedactionRect(rect image.Rectangle, scale float64, bounds image.Rectangle) image.Rectangle {
	if scale <= 0 {
		scale = defaultRedactionBoxScale
	}
	if scale == 1 {
		return rect.Intersect(bounds)
	}

	newWidth := math.Max(1, math.Round(float64(rect.Dx())*scale))
	newHeight := math.Max(1, math.Round(float64(rect.Dy())*scale))
	centerX := float64(rect.Min.X+rect.Max.X) / 2
	centerY := float64(rect.Min.Y+rect.Max.Y) / 2

	scaled := image.Rect(
		int(math.Floor(centerX-newWidth/2)),
		int(math.Floor(centerY-newHeight/2)),
		int(math.Ceil(centerX+newWidth/2)),
		int(math.Ceil(centerY+newHeight/2)),
	)
	return scaled.Intersect(bounds)
}

func (p *paranoidTracker) pruneTracks(frameIndex, lingerFrames int) {
	active := p.activeTracks[:0]
	for _, t := range p.activeTracks {
		if frameIndex-t.LastSeen <= lingerFrames {
			active = append(active, t)
		}
	}
	// Zero out the remainder of the underlying array to prevent memory leaks
	for i := len(active); i < len(p.activeTracks); i++ {
		p.activeTracks[i] = nil
	}
	p.activeTracks = active
}

func redactFace(img *image.RGBA, rect image.Rectangle, style string, strength int) {
	// Clip rect to image bounds to prevent panics
	rect = rect.Intersect(img.Bounds())
	if rect.Empty() {
		return
	}

	switch style {
	case "black":
		stride := img.Stride
		pix := img.Pix
		imgMinX, imgMinY := img.Rect.Min.X, img.Rect.Min.Y
		for y := rect.Min.Y; y < rect.Max.Y; y++ {
			rowStart := (y-imgMinY)*stride + (rect.Min.X-imgMinX)*4
			for x := 0; x < rect.Dx(); x++ {
				off := rowStart + x*4
				pix[off] = 0
				pix[off+1] = 0
				pix[off+2] = 0
				pix[off+3] = 255
			}
		}

	case "secure":
		// "Secure Pixelation": Grab colors from the immediate border and fill with the average.
		// This blends the redaction into the background.
		var r, g, b, count uint64
		stride := img.Stride
		pix := img.Pix
		imgMinX, imgMinY := img.Rect.Min.X, img.Rect.Min.Y

		for x := rect.Min.X; x < rect.Max.X; x++ {
			if edgeY := rect.Min.Y - 1; edgeY >= img.Bounds().Min.Y { // Top
				off := (edgeY-imgMinY)*stride + (x-imgMinX)*4
				r += uint64(pix[off])
				g += uint64(pix[off+1])
				b += uint64(pix[off+2])
				count++
			}
			if edgeY := rect.Max.Y; edgeY < img.Bounds().Max.Y { // Bottom
				off := (edgeY-imgMinY)*stride + (x-imgMinX)*4
				r += uint64(pix[off])
				g += uint64(pix[off+1])
				b += uint64(pix[off+2])
				count++
			}
		}
		for y := rect.Min.Y; y < rect.Max.Y; y++ {
			if edgeX := rect.Min.X - 1; edgeX >= img.Bounds().Min.X { // Left
				off := (y-imgMinY)*stride + (edgeX-imgMinX)*4
				r += uint64(pix[off])
				g += uint64(pix[off+1])
				b += uint64(pix[off+2])
				count++
			}
			if edgeX := rect.Max.X; edgeX < img.Bounds().Max.X { // Right
				off := (y-imgMinY)*stride + (edgeX-imgMinX)*4
				r += uint64(pix[off])
				g += uint64(pix[off+1])
				b += uint64(pix[off+2])
				count++
			}
		}

		var fr, fg, fb uint8 = 0, 0, 0
		if count > 0 {
			fr, fg, fb = uint8(r/count), uint8(g/count), uint8(b/count)
		}

		for y := rect.Min.Y; y < rect.Max.Y; y++ {
			rowStart := (y-imgMinY)*stride + (rect.Min.X-imgMinX)*4
			for x := 0; x < rect.Dx(); x++ {
				off := rowStart + x*4
				pix[off] = fr
				pix[off+1] = fg
				pix[off+2] = fb
				pix[off+3] = 255
			}
		}

	case "gauss":
		// A correct, if naive, implementation of a separable box blur.
		// The previous sliding-window optimization was flawed at the edges, creating artifacts.
		// Given that the radius is typically small, this O(N*R) approach is acceptable for correctness.
		radius := strength
		if radius < 1 {
			radius = 1
		}

		w, h := rect.Dx(), rect.Dy()

		neededSize := w * h * 4
		bufPtr := blurBufferPool.Get().([]uint8)

		if cap(bufPtr) < neededSize {
			bufPtr = make([]uint8, 0, neededSize)
		}
		defer blurBufferPool.Put(bufPtr[:0])

		buf := bufPtr[:neededSize]

		stride := img.Stride
		pix := img.Pix
		minX, minY := rect.Min.X, rect.Min.Y
		imgMinX, imgMinY := img.Rect.Min.X, img.Rect.Min.Y

		// Horizontal Pass: Read from Image -> Write to Buffer
		for y := 0; y < h; y++ {
			rowStart := (minY + y - imgMinY) * stride
			bufRowStart := y * w * 4
			for x := 0; x < w; x++ {
				var rSum, gSum, bSum uint32
				var count int
				for k := -radius; k <= radius; k++ {
					px := x + k
					if px >= 0 && px < w {
						off := rowStart + (minX+px-imgMinX)*4
						rSum += uint32(pix[off])
						gSum += uint32(pix[off+1])
						bSum += uint32(pix[off+2])
						count++
					}
				}

				bufOff := bufRowStart + x*4
				buf[bufOff] = uint8(rSum / uint32(count))
				buf[bufOff+1] = uint8(gSum / uint32(count))
				buf[bufOff+2] = uint8(bSum / uint32(count))
				buf[bufOff+3] = 255
			}
		}

		// Vertical Pass: Read from Buffer -> Write to Image
		for y := 0; y < h; y++ {
			dstRowOff := (minY + y - imgMinY) * stride
			for x := 0; x < w; x++ {
				var rSum, gSum, bSum uint32
				var count int
				for k := -radius; k <= radius; k++ {
					py := y + k
					if py >= 0 && py < h {
						off := py*w*4 + x*4
						rSum += uint32(buf[off])
						gSum += uint32(buf[off+1])
						bSum += uint32(buf[off+2])
						count++
					}
				}

				dstOff := dstRowOff + (minX+x-imgMinX)*4
				pix[dstOff] = uint8(rSum / uint32(count))
				pix[dstOff+1] = uint8(gSum / uint32(count))
				pix[dstOff+2] = uint8(bSum / uint32(count))
				pix[dstOff+3] = 255
			}
		}

	case "pixel":
		fallthrough
	default:
		blockSize := strength
		if blockSize < 1 {
			blockSize = 1
		}
		stride := img.Stride
		pix := img.Pix
		imgMinX, imgMinY := img.Rect.Min.X, img.Rect.Min.Y

		for y := rect.Min.Y; y < rect.Max.Y; y += blockSize {
			for x := rect.Min.X; x < rect.Max.X; x += blockSize {
				srcOff := (y-imgMinY)*stride + (x-imgMinX)*4
				r, g, b, a := pix[srcOff], pix[srcOff+1], pix[srcOff+2], pix[srcOff+3]

				x2 := x + blockSize
				if x2 > rect.Max.X {
					x2 = rect.Max.X
				}
				y2 := y + blockSize
				if y2 > rect.Max.Y {
					y2 = rect.Max.Y
				}

				for by := y; by < y2; by++ {
					rowStart := (by - imgMinY) * stride
					for bx := x; bx < x2; bx++ {
						dstOff := rowStart + (bx-imgMinX)*4
						pix[dstOff] = r
						pix[dstOff+1] = g
						pix[dstOff+2] = b
						pix[dstOff+3] = a
					}
				}
			}
		}
	}
}

func validateRedactFlags(cmd *cobra.Command, opts *Options) error {
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

	mode := effectiveRedactMode(cmd)
	if mode != "blur-all" && mode != "targeted" {
		return fmt.Errorf("invalid mode '%s'. Must be 'blur-all' or 'targeted'", mode)
	}

	validStyles := map[string]bool{"pixel": true, "black": true, "gauss": true, "secure": true}
	if !validStyles[redactStyle] {
		return fmt.Errorf("invalid style '%s'. Must be one of: pixel, black, gauss, secure", redactStyle)
	}

	targets := strings.TrimSpace(redactTargets)
	if cmd != nil {
		if resolvedTargets, err := cmd.Flags().GetString("target"); err == nil {
			targets = strings.TrimSpace(resolvedTargets)
		}
	}

	if mode == "targeted" && targets == "" {
		return fmt.Errorf("targeted mode requires --target list of IDs")
	}
	if targets != "" && cmd != nil && cmd.Flags().Changed("mode") && mode == "blur-all" {
		return fmt.Errorf("--target cannot be used with explicit --mode blur-all")
	}

	if _, err := time.ParseDuration(redactLinger); err != nil {
		return fmt.Errorf("invalid linger format: %w (use '1s', '500ms')", err)
	}

	if opts.NumEngines < 1 {
		opts.NumEngines = 1
	}

	if opts.MatchThreshold < 0 || opts.MatchThreshold > 1.0 {
		return fmt.Errorf("invalid match threshold: must be between 0.0 and 1.0, got %f", opts.MatchThreshold)
	}

	if opts.DetectionThreshold < 0 || opts.DetectionThreshold > 1.0 {
		return fmt.Errorf("invalid detection threshold: must be between 0.0 and 1.0, got %f", opts.DetectionThreshold)
	}

	if opts.BlurStrength < 0 {
		return fmt.Errorf("invalid strength: must be non-negative, got %d", opts.BlurStrength)
	}
	boxScaleExplicit := false
	if cmd != nil {
		boxScaleExplicit = cmd.Flags().Changed("box-scale")
	}
	if opts.BoxScale == 0 && !boxScaleExplicit {
		opts.BoxScale = defaultRedactionBoxScale
	}
	if opts.BoxScale <= 0 {
		return fmt.Errorf("invalid box-scale: must be greater than 0, got %f", opts.BoxScale)
	}

	if _, err := time.ParseDuration(opts.WorkerTimeout); err != nil {
		return fmt.Errorf("invalid worker-timeout format: %w (use '30s', '1m')", err)
	}

	return nil
}
