package worker

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"time"

	"github.com/andresmejia3/sentinel/internal/types"
	"github.com/andresmejia3/sentinel/internal/utils" // Using the SafeCommand wrapper
)

// baseWorker manages the common external Python process logic (pipes, handshake).
type baseWorker struct {
	ID          int
	Cmd         *utils.SafeCommand
	Stdin       io.WriteCloser
	DataPipe    io.ReadCloser
	readBuf     []byte // Reusable buffer to reduce GC pressure
	readTimeout time.Duration
}

// ScanWorker is a worker specialized for indexing (thumbnails, quality scores).
type ScanWorker struct {
	*baseWorker
}

// RedactWorker is a worker specialized for redaction (optimized protocol, no thumbnails).
type RedactWorker struct {
	*baseWorker
}

// ScanConfig holds configuration for the ScanWorker.
type ScanConfig struct {
	Debug              bool
	DetectionThreshold float64
	ReadTimeout        time.Duration
	QualityStrategy    string
}

// RedactConfig holds configuration for the RedactWorker.
type RedactConfig struct {
	Debug              bool
	DetectionThreshold float64
	ReadTimeout        time.Duration
	InferenceMode      string // "full" or "detection-only"
	RawWidth           int    // If > 0, input is raw RGBA
	RawHeight          int
}

// newBaseWorker spawns a Python process and sets up the IPC pipes.
func newBaseWorker(ctx context.Context, id int, script string, args []string, readTimeout time.Duration) (*baseWorker, error) {
	resolvedScript := resolveScriptPath(script)
	fullArgs := append([]string{"-u", resolvedScript}, args...)

	py := utils.NewSafeCommand(ctx, "python3", fullArgs...)

	// Create a side-channel pipe (FD 3) for clean data transfer
	r, w, err := os.Pipe()
	if err != nil {
		return nil, fmt.Errorf("failed to create pipe: %w", err)
	}
	// Pass the write-end to the child process. It will appear as FD 3.
	py.Cmd.ExtraFiles = []*os.File{w}

	py.Cmd.Stderr = py.Stderr // Capture stderr for crash logs
	py.Cmd.Stdout = nil       // Discard stdout to silence library noise

	stdin, err := py.StdinPipe()
	if err != nil {
		w.Close() // Prevent FD leak
		r.Close()
		return nil, fmt.Errorf("failed to create stdin pipe: %w", err)
	}

	if err := py.Start(); err != nil {
		w.Close() // Close write end if start fails
		r.Close()
		return nil, fmt.Errorf("worker %d failed to start: %w", id, err)
	}

	// Close the write-end in the parent so only the child holds it
	w.Close()

	// --- Handshake ---
	// Wait for the "READY" signal, respecting context cancellation to prevent hangs.
	handshakeResult := make(chan error, 1)
	go func() {
		readyBuf := make([]byte, 5)
		if _, err := io.ReadFull(r, readyBuf); err != nil {
			handshakeResult <- fmt.Errorf("failed handshake read: %w", err)
			return
		}
		if string(readyBuf) != "READY" {
			handshakeResult <- fmt.Errorf("bad handshake: expected 'READY', got '%s'", string(readyBuf))
			return
		}
		handshakeResult <- nil
	}()

	select {
	case err := <-handshakeResult:
		if err != nil {
			r.Close()
			py.Cmd.Wait() // Wait for process to exit to get logs
			return nil, fmt.Errorf("worker %d handshake failed: %w\nLogs:\n%s", id, err, py.Stderr.String())
		}
	case <-ctx.Done():
		r.Close()
		py.Cmd.Wait() // Wait for the process to exit (cleaned up by CommandContext) to avoid zombies
		return nil, fmt.Errorf("worker %d handshake cancelled: %w", id, ctx.Err())
	}

	return &baseWorker{
		ID:          id,
		Cmd:         py,
		Stdin:       stdin,
		DataPipe:    r,
		readBuf:     make([]byte, 0, 64*1024), // Pre-allocate 64KB to minimize initial resizing
		readTimeout: readTimeout,
	}, nil
}

// resolveScriptPath attempts to find the python script in multiple locations.
// 1. Relative to CWD (Dev mode)
// 2. Relative to Executable (Binary mode)
// 3. Fixed /app/ location (Docker mode)
func resolveScriptPath(script string) string {
	// 1. Check CWD
	if _, err := os.Stat(script); err == nil {
		return script
	}

	// 2. Check relative to Executable
	if ex, err := os.Executable(); err == nil {
		exPath := filepath.Join(filepath.Dir(ex), script)
		if _, err := os.Stat(exPath); err == nil {
			return exPath
		}
	}

	// 3. Check Docker default
	dockerPath := filepath.Join("/app", script)
	if _, err := os.Stat(dockerPath); err == nil {
		return dockerPath
	}

	return script
}

func NewPythonScanWorker(ctx context.Context, id int, cfg ScanConfig) (*ScanWorker, error) {
	args := []string{}
	if cfg.Debug {
		args = append(args, "--debug")
	}
	args = append(args, "--detection-threshold", fmt.Sprintf("%f", cfg.DetectionThreshold))
	if cfg.QualityStrategy != "" {
		args = append(args, "--quality-strategy", cfg.QualityStrategy)
	}

	base, err := newBaseWorker(ctx, id, "python/scan_worker.py", args, cfg.ReadTimeout)
	if err != nil {
		return nil, err
	}
	return &ScanWorker{baseWorker: base}, nil
}

func NewPythonRedactWorker(ctx context.Context, id int, cfg RedactConfig) (*RedactWorker, error) {
	args := []string{}
	if cfg.Debug {
		args = append(args, "--debug")
	}
	args = append(args, "--detection-threshold", fmt.Sprintf("%f", cfg.DetectionThreshold))
	if cfg.InferenceMode != "" {
		args = append(args, "--inference-mode", cfg.InferenceMode)
	}
	if cfg.RawWidth > 0 && cfg.RawHeight > 0 {
		args = append(args, "--raw-width", fmt.Sprintf("%d", cfg.RawWidth))
		args = append(args, "--raw-height", fmt.Sprintf("%d", cfg.RawHeight))
	}

	base, err := newBaseWorker(ctx, id, "python/redact_worker.py", args, cfg.ReadTimeout)
	if err != nil {
		return nil, err
	}
	return &RedactWorker{baseWorker: base}, nil
}

// sendFrame writes the frame data to the worker's stdin with a length header.
func (w *baseWorker) sendFrame(data []byte) error {
	var header [4]byte
	binary.BigEndian.PutUint32(header[:], uint32(len(data)))
	if _, err := w.Stdin.Write(header[:]); err != nil {
		return err
	}
	if _, err := w.Stdin.Write(data); err != nil {
		return err
	}
	return nil
}

// readResponse reads the length-prefixed response from the worker.
func (w *baseWorker) readResponse() ([]byte, error) {
	if f, ok := w.DataPipe.(*os.File); ok && w.readTimeout > 0 {
		f.SetReadDeadline(time.Now().Add(w.readTimeout))
		defer f.SetReadDeadline(time.Time{})
	}

	var header [4]byte
	if _, err := io.ReadFull(w.DataPipe, header[:]); err != nil {
		return nil, err // This is where we catch the "ModuleNotFoundError" crash
	}

	respLen := binary.BigEndian.Uint32(header[:])
	if respLen > 1<<30 {
		// Safety check: Max 1GB response (supports extreme crowd scenes) so we don't
		// allocate enormous amounts in the following make call due to a corrupt bit
		return nil, fmt.Errorf("worker returned oversized response: %d bytes", respLen)
	}
	if respLen < 5 {
		// Minimum payload: Status (1 byte) + NumFaces (4 bytes)
		return nil, fmt.Errorf("worker returned invalid response length: %d bytes (min 5)", respLen)
	}

	// Optimization: Use a growth strategy (2x) to avoid frequent re-allocations when frame sizes fluctuate.
	needed := int(respLen)
	if cap(w.readBuf) < needed {
		newCap := cap(w.readBuf) * 2
		if newCap < needed {
			newCap = needed
		}
		w.readBuf = make([]byte, newCap)
	} else if cap(w.readBuf) > 64*1024*1024 && needed < 1024*1024 {
		// Shrink if excessively large (>64MB) and we only need a small amount (<1MB)
		// Optimization: Shrink to a baseline (1MB) instead of 'needed' to prevent thrashing
		w.readBuf = make([]byte, 1024*1024)
	}
	w.readBuf = w.readBuf[:needed]

	if _, err := io.ReadFull(w.DataPipe, w.readBuf); err != nil {
		return nil, err
	}

	return w.readBuf, nil
}

func (w *ScanWorker) ProcessScanFrame(data []byte) ([]types.FaceResult, error) {
	if err := w.sendFrame(data); err != nil {
		return nil, err
	}

	resp, err := w.readResponse()
	if err != nil {
		return nil, err
	}

	cursor := 0

	status := resp[cursor]
	cursor++

	if status == 1 {
		msgLen := binary.BigEndian.Uint32(resp[cursor:])
		cursor += 4
		if msgLen > 1<<20 { // Safety check: Max 1MB error message
			return nil, fmt.Errorf("python worker returned oversized error message: %d bytes", msgLen)
		}
		msg := resp[cursor : cursor+int(msgLen)]
		return nil, fmt.Errorf("python worker error: %s", string(msg))
	}

	numFaces := binary.BigEndian.Uint32(resp[cursor:])
	cursor += 4

	results := make([]types.FaceResult, numFaces)

	// OPTIMIZATION: Allocate ONE contiguous backing array for all vectors in this frame.
	// CRITICAL FIX: We cannot reuse a persistent buffer (w.vecBuf) here because these slices
	// escape to the results channel and are read asynchronously by the aggregator.
	allVecs := make([]float64, int(numFaces)*512)

	for i := 0; i < int(numFaces); i++ {
		// [Box: 16B]
		x1 := int32(binary.BigEndian.Uint32(resp[cursor:]))
		y1 := int32(binary.BigEndian.Uint32(resp[cursor+4:]))
		x2 := int32(binary.BigEndian.Uint32(resp[cursor+8:]))
		y2 := int32(binary.BigEndian.Uint32(resp[cursor+12:]))
		cursor += 16

		// [Vec: 2048B]
		// Slice from our pre-allocated backing array
		vec64 := allVecs[i*512 : (i+1)*512]
		for j := 0; j < 512; j++ {
			bits := binary.BigEndian.Uint32(resp[cursor:])
			vec64[j] = float64(math.Float32frombits(bits))
			cursor += 4
		}

		// [Quality: 4B]
		qualityBits := binary.BigEndian.Uint32(resp[cursor:])
		quality := math.Float32frombits(qualityBits)
		cursor += 4

		// [ImgLen: 4B] + [ImgData]
		imgLen := binary.BigEndian.Uint32(resp[cursor:])
		cursor += 4

		// Copy image data to a new slice to prevent corruption when readBuf is reused
		imgData := make([]byte, imgLen)
		copy(imgData, resp[cursor:cursor+int(imgLen)])
		cursor += int(imgLen)

		results[i] = types.FaceResult{
			Loc:     []int{int(x1), int(y1), int(x2), int(y2)},
			Vec:     vec64,
			Quality: float64(quality),
			Thumb:   imgData,
		}
	}

	return results, nil
}

func (w *RedactWorker) ProcessRedactFrame(data []byte) ([]types.FaceResult, error) {
	if err := w.sendFrame(data); err != nil {
		return nil, err
	}

	resp, err := w.readResponse()
	if err != nil {
		return nil, err
	}

	cursor := 0

	status := resp[cursor]
	cursor++

	if status == 1 {
		msgLen := binary.BigEndian.Uint32(resp[cursor:])
		cursor += 4
		if msgLen > 1<<20 {
			return nil, fmt.Errorf("python worker returned oversized error message: %d bytes", msgLen)
		}
		msg := resp[cursor : cursor+int(msgLen)]
		return nil, fmt.Errorf("python worker error: %s", string(msg))
	}

	numFaces := binary.BigEndian.Uint32(resp[cursor:])
	cursor += 4

	results := make([]types.FaceResult, numFaces)
	allVecs := make([]float64, int(numFaces)*512)

	for i := 0; i < int(numFaces); i++ {
		// [Box: 16B]
		x1 := int32(binary.BigEndian.Uint32(resp[cursor:]))
		y1 := int32(binary.BigEndian.Uint32(resp[cursor+4:]))
		x2 := int32(binary.BigEndian.Uint32(resp[cursor+8:]))
		y2 := int32(binary.BigEndian.Uint32(resp[cursor+12:]))
		cursor += 16

		// [Vec: 2048B]
		vec64 := allVecs[i*512 : (i+1)*512]
		for j := 0; j < 512; j++ {
			bits := binary.BigEndian.Uint32(resp[cursor:])
			vec64[j] = float64(math.Float32frombits(bits))
			cursor += 4
		}

		// No Quality, No Thumb
		results[i] = types.FaceResult{
			Loc:     []int{int(x1), int(y1), int(x2), int(y2)},
			Vec:     vec64,
			Quality: 0,
			Thumb:   nil,
		}
	}

	return results, nil
}

// Close cleans up the worker resources, closing pipes and waiting for the process to exit.
func (w *baseWorker) Close() {
	w.Stdin.Close()
	w.DataPipe.Close()
	w.Cmd.Wait()
}
