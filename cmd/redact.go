package cmd

import (
	"context"
	"fmt"
	"image"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/andresmejia3/sentinel/internal/types"
	"github.com/andresmejia3/sentinel/internal/utils"
	"github.com/andresmejia3/sentinel/internal/worker"
	"github.com/schollz/progressbar/v3"
	"github.com/spf13/cobra"
)

var (
	redactOpts     Options
	redactOutput   string
	redactMode     string
	redactTargets  string
	redactStyle    string
	redactLinger   string
	redactParanoid bool
)

var redactCmd = &cobra.Command{
	Use:   "redact",
	Short: "Redact faces in a video based on detection or identity",
	RunE: func(cmd *cobra.Command, args []string) error {
		cmd.SilenceUsage = true
		return runRedact(cmd.Context(), redactOpts)
	},
}

func init() {
	redactCmd.Flags().StringVarP(&redactOpts.InputPath, "input", "i", "", "Path to input video")
	redactCmd.Flags().StringVarP(&redactOutput, "output", "o", "redacted.mp4", "Path to output video")
	redactCmd.Flags().StringVarP(&redactMode, "mode", "m", "blur-all", "Redaction mode: blur-all, targeted")
	redactCmd.Flags().StringVar(&redactTargets, "target", "", "Comma-separated list of Identity IDs to redact (for targeted mode)")
	redactCmd.Flags().StringVar(&redactStyle, "style", "black", "Redaction style: pixel, black, gauss, secure")
	redactCmd.Flags().StringVar(&redactLinger, "linger", "1s", "How long to keep blurring after a targeted face is lost")
	redactCmd.Flags().BoolVar(&redactParanoid, "paranoid", false, "Enable paranoid mode: blur ALL faces if a targeted face is lost")

	redactCmd.Flags().IntVarP(&redactOpts.NumEngines, "engines", "e", 1, "Number of parallel engine workers")
	redactCmd.Flags().Float64VarP(&redactOpts.MatchThreshold, "threshold", "t", 0.6, "Face matching threshold")
	redactCmd.Flags().Float64VarP(&redactOpts.DetectionThreshold, "detection-threshold", "D", 0.5, "Face detection confidence threshold")
	redactCmd.Flags().IntVarP(&redactOpts.BlurStrength, "strength", "s", 15, "Pixelation block size (higher = more blocky)")
	redactCmd.Flags().StringVar(&redactOpts.WorkerTimeout, "worker-timeout", "30s", "Timeout for a worker to process a single frame")

	redactCmd.MarkFlagRequired("input")
	rootCmd.AddCommand(redactCmd)
}

// blurBufferPool recycles scratch buffers for the Gaussian blur.
var blurBufferPool = sync.Pool{
	New: func() interface{} { return make([]uint8, 0, 1024*1024) }, // Start with 1MB capacity
}

// colSumsPool recycles column accumulators for the Gaussian blur.
var colSumsPool = sync.Pool{
	New: func() interface{} { return make([]uint32, 0, 1024) },
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
	targetVecs   map[int][]float64 // Optimization 1: In-memory embeddings
	activeTracks []*redactionTrack
	opts         *Options
}

func runRedact(ctx context.Context, opts Options) error {
	// Create a cancellable context to ensure all child processes (FFmpeg, Python)
	// are killed immediately if this function returns early (e.g. on error).
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	if err := validateRedactFlags(&opts); err != nil {
		return err
	}

	// Safety Check: Prevent overwriting input file which causes corruption
	inAbs, _ := filepath.Abs(opts.InputPath)
	outAbs, _ := filepath.Abs(redactOutput)
	if inAbs == outAbs {
		return fmt.Errorf("input and output paths must be different to prevent file corruption")
	}

	targetIDs := []int{}
	if redactMode == "targeted" {
		for _, s := range strings.Split(redactTargets, ",") {
			id, err := strconv.Atoi(strings.TrimSpace(s))
			if err == nil {
				targetIDs = append(targetIDs, id)
			}
		}
	}

	fps, err := utils.GetVideoFPS(ctx, opts.InputPath)
	if err != nil {
		utils.ShowError("Failed to determine video FPS", err, nil)
		return err
	}
	width, height, err := utils.GetVideoDimensions(ctx, opts.InputPath)
	if err != nil {
		utils.ShowError("Failed to determine video dimensions", err, nil)
		return err
	}
	totalFrames := utils.GetTotalFrames(ctx, opts.InputPath)

	taskChan := make(chan types.FrameTask, opts.NumEngines)
	resultsChan := make(chan redactResult, opts.NumEngines*2)
	errChan := make(chan error, opts.NumEngines+2)

	var wg sync.WaitGroup
	readyChan := make(chan bool, opts.NumEngines)

	// Optimization 1: Load target embeddings into memory
	targetVecs, err := DB.GetIdentityVectors(ctx, targetIDs)
	if err != nil {
		utils.ShowError("Failed to load target embeddings", err, nil)
		return err
	}

	lingerDuration, _ := time.ParseDuration(redactLinger)
	lingerFrames := int(lingerDuration.Seconds() * fps)
	tracker := &paranoidTracker{
		targetVecs:   targetVecs,
		activeTracks: make([]*redactionTrack, 0),
		opts:         &opts,
	}

	workerTimeout, _ := time.ParseDuration(opts.WorkerTimeout)

	for i := 0; i < opts.NumEngines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()

			inferenceMode := "full"
			if redactMode == "blur-all" {
				inferenceMode = "detection-only"
			}

			cfg := worker.RedactConfig{
				DetectionThreshold: opts.DetectionThreshold,
				ReadTimeout:        workerTimeout,
				InferenceMode:      inferenceMode,
				RawWidth:           width,
				RawHeight:          height,
			}
			w, err := worker.NewPythonRedactWorker(ctx, id, cfg)
			if err != nil {
				utils.ShowError("Worker startup failed", err, nil)
				select {
				case errChan <- err:
				default:
				}
				return
			}
			defer w.Close()
			readyChan <- true

			for task := range taskChan {
				faces, err := w.ProcessRedactFrame(task.Data)
				if err != nil {
					utils.ShowError("Python crashed", err, w.Cmd)
					select {
					case errChan <- err:
					default:
					}
					return
				}
				select {
				case resultsChan <- redactResult{Index: task.Index, Data: task.Data, Faces: faces}:
				case <-ctx.Done():
					return
				}
			}
		}(i)
	}

	// Wait for workers to be ready
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

	decoder := utils.NewFFmpegRawDecoder(ctx, opts.InputPath)
	decoderOut, err := decoder.StdoutPipe()
	if err != nil {
		utils.ShowError("Failed to create decoder pipe", err, nil)
		return err
	}
	if err := decoder.Start(); err != nil {
		utils.ShowError("Failed to start decoder", err, nil)
		return err
	}

	encoder := utils.NewFFmpegEncoder(ctx, redactOutput, fps, width, height)
	encoderIn, err := encoder.StdinPipe()
	if err != nil {
		utils.ShowError("Failed to create encoder pipe", err, nil)
		return err
	}
	if err := encoder.Start(); err != nil {
		utils.ShowError("Failed to start encoder", err, nil)
		return err
	}

	go func() {
		frameSize := width * height * 4
		idx := 0
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

			select {
			case taskChan <- types.FrameTask{Index: idx, Data: buf}:
				idx++
			case <-ctx.Done():
				return
			}
		}
		close(taskChan)
	}()

	buffer := make(map[int]redactResult)
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

	go func() {
		wg.Wait()
		close(resultsChan)
	}()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case err := <-errChan:
			return err
		case res, ok := <-resultsChan:
			if !ok {
				goto Flush
			}
			buffer[res.Index] = res

			for {
				frame, ok := buffer[nextFrame]
				if !ok {
					break
				}
				delete(buffer, nextFrame)

				outData, err := tracker.apply(ctx, frame.Data, width, height, nextFrame, frame.Faces, redactMode, redactStyle, redactParanoid, lingerFrames)
				if err != nil {
					utils.ShowError("Redaction failed", err, nil)
					return err
				}
				if _, err := encoderIn.Write(outData); err != nil {
					return err
				}

				// Release buffer back to pool
				frameBufferPool.Put(frame.Data)

				bar.Add(1)
				nextFrame++
			}
		}
	}

Flush:
	encoderIn.Close()
	if err := encoder.Wait(); err != nil {
		utils.ShowError("Encoder process failed", err, nil)
		return err
	}
	if err := decoder.Wait(); err != nil {
		utils.ShowError("Decoder process failed", err, nil)
		return err
	}
	return nil
}

func (p *paranoidTracker) apply(ctx context.Context, imgData []byte, width, height, frameIndex int, faces []types.FaceResult, mode, style string, paranoid bool, lingerFrames int) ([]byte, error) {
	// Zero-Copy: Wrap the raw bytes in an image.RGBA struct
	m := &image.RGBA{
		Pix:    imgData,
		Stride: width * 4,
		Rect:   image.Rect(0, 0, width, height),
	}
	strength := p.opts.BlurStrength

	// Optimization 2: Copy master list to a "missing" set.
	// We remove IDs as we find them. If any remain, paranoid mode triggers.
	missingTargets := make(map[int]bool, len(p.targetVecs))
	if mode == "targeted" {
		for id := range p.targetVecs {
			missingTargets[id] = true
		}
	}

	// Map to track which face index corresponds to which target ID (for blurring later)
	faceIdentities := make([]int, len(faces))
	for i := range faceIdentities {
		faceIdentities[i] = -1
	}

	// 1. Identify faces using In-Memory Embeddings
	for i, face := range faces {
		if mode == "targeted" {
			bestID := -1
			minDist := p.opts.MatchThreshold

			// Check against loaded targets
			for tID, tVec := range p.targetVecs {
				dist := cosineDist(face.Vec, tVec)
				if dist < minDist {
					minDist = dist
					bestID = tID
				}
			}

			if bestID != -1 {
				faceIdentities[i] = bestID
				delete(missingTargets, bestID) // Found it! Remove from missing list.

				// Update Linger Track
				p.updateTrack(bestID, frameIndex, image.Rect(face.Loc[0], face.Loc[1], face.Loc[2], face.Loc[3]))
			}
		}
	}

	// 2. Paranoid Check
	isParanoidActive := false
	if paranoid && mode == "targeted" && len(missingTargets) > 0 {
		isParanoidActive = true
	}

	// 3. Apply Redaction
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
			rect := image.Rect(face.Loc[0], face.Loc[1], face.Loc[2], face.Loc[3])
			redactFace(m, rect, style, strength)
		}
	}

	// 4. Apply Linger (only in targeted mode)
	// If paranoid is active, we are blurring all detected faces.
	// However, linger covers *undetected* faces (lost tracking), so we still apply it.
	if mode == "targeted" {
		// We iterate active tracks. If a track is in 'missingTargets', it means it wasn't found in this frame.
		for _, track := range p.activeTracks {
			if missingTargets[track.ID] {
				if frameIndex-track.LastSeen <= lingerFrames {
					redactFace(m, track.LastKnown, style, strength)
				}
			}
		}
	}

	// 5. Prune old tracks
	p.pruneTracks(frameIndex, lingerFrames)

	return m.Pix, nil
}

func (p *paranoidTracker) updateTrack(id, frameIndex int, rect image.Rectangle) {
	for _, t := range p.activeTracks {
		if t.ID == id {
			t.LastSeen = frameIndex
			t.LastKnown = rect
			return
		}
	}
	p.activeTracks = append(p.activeTracks, &redactionTrack{
		ID:        id,
		LastSeen:  frameIndex,
		LastKnown: rect,
	})
}

func (p *paranoidTracker) pruneTracks(frameIndex, lingerFrames int) {
	active := p.activeTracks[:0]
	for _, t := range p.activeTracks {
		if frameIndex-t.LastSeen <= lingerFrames {
			active = append(active, t)
		}
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

		// Scan Top & Bottom borders
		for x := rect.Min.X; x < rect.Max.X; x++ {
			if y := rect.Min.Y - 1; y >= img.Bounds().Min.Y { // Top
				off := (y-imgMinY)*stride + (x-imgMinX)*4
				r += uint64(pix[off])
				g += uint64(pix[off+1])
				b += uint64(pix[off+2])
				count++
			}
			if y := rect.Max.Y; y < img.Bounds().Max.Y { // Bottom
				off := (y-imgMinY)*stride + (x-imgMinX)*4
				r += uint64(pix[off])
				g += uint64(pix[off+1])
				b += uint64(pix[off+2])
				count++
			}
		}
		// Scan Left & Right borders
		for y := rect.Min.Y; y < rect.Max.Y; y++ {
			if x := rect.Min.X - 1; x >= img.Bounds().Min.X { // Left
				off := (y-imgMinY)*stride + (x-imgMinX)*4
				r += uint64(pix[off])
				g += uint64(pix[off+1])
				b += uint64(pix[off+2])
				count++
			}
			if x := rect.Max.X; x < img.Bounds().Max.X { // Right
				off := (y-imgMinY)*stride + (x-imgMinX)*4
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
		// Separable Box Blur
		// Strength = Kernel Radius. This allows for strong blurs (e.g. radius 20)
		// without the O(N^2) cost of a naive convolution.
		radius := strength
		if radius < 1 {
			radius = 1
		}

		w, h := rect.Dx(), rect.Dy()
		// Clamp radius to prevent looking outside the face bounds
		if radius > w/2 {
			radius = w / 2
		}
		if radius > h/2 {
			radius = h / 2
		}

		// Intermediate buffer for the horizontal pass
		// Optimization: Use a pooled buffer to avoid allocating per-face
		neededSize := w * h * 4
		bufPtr := blurBufferPool.Get().([]uint8)
		if cap(bufPtr) < neededSize {
			bufPtr = make([]uint8, neededSize)
		}
		buf := bufPtr[:neededSize]
		// Ensure we return the buffer to the pool when done
		defer blurBufferPool.Put(bufPtr)

		stride := img.Stride
		pix := img.Pix
		minX, minY := rect.Min.X, rect.Min.Y
		imgMinX, imgMinY := img.Rect.Min.X, img.Rect.Min.Y

		// 1. Horizontal Pass: Read from Image -> Write to Buffer
		for y := 0; y < h; y++ {
			rowStart := (minY + y - imgMinY) * stride
			bufRowStart := y * w * 4

			// Initialize Accumulator
			var rSum, gSum, bSum uint32
			for k := -radius; k <= radius; k++ {
				px := k
				if px < 0 {
					px = 0
				}
				if px >= w {
					px = w - 1
				}
				off := rowStart + (minX+px-imgMinX)*4
				rSum += uint32(pix[off])
				gSum += uint32(pix[off+1])
				bSum += uint32(pix[off+2])
			}

			count := uint32(2*radius + 1)

			for x := 0; x < w; x++ {
				bufOff := bufRowStart + x*4
				buf[bufOff] = uint8(rSum / count)
				buf[bufOff+1] = uint8(gSum / count)
				buf[bufOff+2] = uint8(bSum / count)
				buf[bufOff+3] = 255

				// Slide Window: Subtract leaving pixel, Add entering pixel
				pRemove := x - radius
				if pRemove < 0 {
					pRemove = 0
				}

				pAdd := x + radius + 1
				if pAdd >= w {
					pAdd = w - 1
				}

				offRemove := rowStart + (minX+pRemove-imgMinX)*4
				offAdd := rowStart + (minX+pAdd-imgMinX)*4

				rSum = rSum - uint32(pix[offRemove]) + uint32(pix[offAdd])
				gSum = gSum - uint32(pix[offRemove+1]) + uint32(pix[offAdd+1])
				bSum = bSum - uint32(pix[offRemove+2]) + uint32(pix[offAdd+2])
			}
		}

		// 2. Vertical Pass: Read from Buffer -> Write to Image
		// Optimization: Process ROW-BY-ROW instead of COLUMN-BY-COLUMN to improve CPU cache locality.
		// We maintain running sums for every column in parallel.

		// Use pool to reduce allocation
		neededCols := w * 3
		csPtr := colSumsPool.Get().([]uint32)
		if cap(csPtr) < neededCols {
			csPtr = make([]uint32, neededCols)
		}
		colSums := csPtr[:neededCols]
		// Must zero out recycled buffer
		for i := range colSums {
			colSums[i] = 0
		}
		defer colSumsPool.Put(csPtr)

		// Initialize accumulators for all columns (Kernel centered at y=0)
		for k := -radius; k <= radius; k++ {
			py := k
			if py < 0 {
				py = 0
			}
			if py >= h {
				py = h - 1
			}

			rowOffset := py * w * 4
			for x := 0; x < w; x++ {
				off := rowOffset + x*4
				colSums[x*3] += uint32(buf[off])
				colSums[x*3+1] += uint32(buf[off+1])
				colSums[x*3+2] += uint32(buf[off+2])
			}
		}

		count := uint32(2*radius + 1)

		for y := 0; y < h; y++ {
			dstRowOff := (minY + y - imgMinY) * stride

			for x := 0; x < w; x++ {
				// Apply Blur to current pixel
				dstOff := dstRowOff + (minX+x-imgMinX)*4
				pix[dstOff] = uint8(colSums[x*3] / count)
				pix[dstOff+1] = uint8(colSums[x*3+1] / count)
				pix[dstOff+2] = uint8(colSums[x*3+2] / count)

				// Update Accumulator for next row (y+1)
				pRemove := y - radius
				if pRemove < 0 {
					pRemove = 0
				}

				pAdd := y + radius + 1
				if pAdd >= h {
					pAdd = h - 1
				}

				offRemove := pRemove*w*4 + x*4
				offAdd := pAdd*w*4 + x*4

				colSums[x*3] = colSums[x*3] - uint32(buf[offRemove]) + uint32(buf[offAdd])
				colSums[x*3+1] = colSums[x*3+1] - uint32(buf[offRemove+1]) + uint32(buf[offAdd+1])
				colSums[x*3+2] = colSums[x*3+2] - uint32(buf[offRemove+2]) + uint32(buf[offAdd+2])
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
				// Get color of top-left pixel
				srcOff := (y-imgMinY)*stride + (x-imgMinX)*4
				r, g, b, a := pix[srcOff], pix[srcOff+1], pix[srcOff+2], pix[srcOff+3]

				// Calculate block bounds
				x2 := x + blockSize
				if x2 > rect.Max.X {
					x2 = rect.Max.X
				}
				y2 := y + blockSize
				if y2 > rect.Max.Y {
					y2 = rect.Max.Y
				}

				// Fill block using direct slice access
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

func validateRedactFlags(opts *Options) error {
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

	if redactMode != "blur-all" && redactMode != "targeted" {
		err := fmt.Errorf("invalid mode '%s'. Must be 'blur-all' or 'targeted'", redactMode)
		utils.ShowError("Configuration Error", err, nil)
		return err
	}

	validStyles := map[string]bool{"pixel": true, "black": true, "gauss": true, "secure": true}
	if !validStyles[redactStyle] {
		err := fmt.Errorf("invalid style '%s'. Must be one of: pixel, black, gauss, secure", redactStyle)
		utils.ShowError("Configuration Error", err, nil)
		return err
	}

	if redactMode == "targeted" && redactTargets == "" {
		err := fmt.Errorf("targeted mode requires --target list of IDs")
		utils.ShowError("Configuration Error", err, nil)
		return err
	}

	if _, err := time.ParseDuration(redactLinger); err != nil {
		utils.ShowError("Invalid linger format (use '1s', '500ms')", err, nil)
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

	if opts.DetectionThreshold <= 0 || opts.DetectionThreshold > 1.0 {
		err := fmt.Errorf("must be between 0.0 and 1.0, got %f", opts.DetectionThreshold)
		utils.ShowError("Invalid detection threshold", err, nil)
		return err
	}

	if _, err := time.ParseDuration(opts.WorkerTimeout); err != nil {
		utils.ShowError("Invalid worker-timeout format (use '30s', '1m')", err, nil)
		return err
	}

	return nil
}
