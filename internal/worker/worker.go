package worker

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"time"

	"github.com/andresmejia3/sentinel/internal/types"
	"github.com/andresmejia3/sentinel/internal/utils" // Using the SafeCommand wrapper
)

// PythonWorker manages the external Python process for neural inference.
type PythonWorker struct {
	ID          int
	Cmd         *utils.SafeCommand
	Stdin       io.WriteCloser
	DataPipe    io.ReadCloser
	readBuf     []byte // Reusable buffer to reduce GC pressure
	readTimeout time.Duration
}

// Config holds the configuration for the Python worker process.
type Config struct {
	Debug              bool
	DetectionThreshold float64
	ReadTimeout        time.Duration
	QualityStrategy    string
}

// NewPythonWorker spawns a new Python process and sets up the IPC pipes (Stdin + Side-channel).
func NewPythonWorker(ctx context.Context, id int, cfg Config) (*PythonWorker, error) {
	args := []string{"-u", "python/worker.py"}
	if cfg.Debug {
		args = append(args, "--debug")
	}
	args = append(args, "--detection-threshold", fmt.Sprintf("%f", cfg.DetectionThreshold))
	if cfg.QualityStrategy != "" {
		args = append(args, "--quality-strategy", cfg.QualityStrategy)
	}
	// 1. Initialize the SafeCommand we built
	py := utils.NewSafeCommand(ctx, "python3", args...)

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
		r.Close() // Close read-end too!
		return nil, fmt.Errorf("failed to create stdin pipe: %w", err)
	}

	if err := py.Start(); err != nil {
		w.Close() // Close write end if start fails
		r.Close() // Close read-end too!
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
		r.Close()     // Close the read pipe
		py.Cmd.Wait() // Wait for the process to exit (cleaned up by CommandContext) to avoid zombies
		return nil, fmt.Errorf("worker %d handshake cancelled: %w", id, ctx.Err())
	}

	return &PythonWorker{
		ID:          id,
		Cmd:         py,
		Stdin:       stdin,
		DataPipe:    r,
		readBuf:     make([]byte, 0, 64*1024), // Pre-allocate 64KB to minimize initial resizing
		readTimeout: cfg.ReadTimeout,
	}, nil
}

// ProcessFrame sends a frame to the worker via Stdin and waits for the JSON response via the DataPipe.
// It handles the binary protocol: [4-byte Length][Payload].
func (w *PythonWorker) ProcessFrame(data []byte) ([]types.FaceResult, error) {
	// Protocol: [Length][Data]
	// We perform two writes to avoid allocating a massive buffer just to prepend 4 bytes.
	// The syscall overhead is negligible compared to the memory allocation/copy cost.
	var header [4]byte
	binary.BigEndian.PutUint32(header[:], uint32(len(data)))
	if _, err := w.Stdin.Write(header[:]); err != nil {
		return nil, err
	}
	if _, err := w.Stdin.Write(data); err != nil {
		return nil, err
	}

	// Read Result
	// Now we read from our clean DataPipe, so no Magic Byte is needed.
	// Set a deadline to prevent hanging forever if Python freezes.
	if f, ok := w.DataPipe.(*os.File); ok && w.readTimeout > 0 {
		f.SetReadDeadline(time.Now().Add(w.readTimeout))
		defer f.SetReadDeadline(time.Time{})
	}

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
		return nil, fmt.Errorf("worker returned invalid response length: %d bytes (min 5)", respLen)
	}

	// Resize reusable buffer
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

	// --- Parse Binary Payload ---
	// We use a cursor and direct slice access to avoid the overhead of bytes.Reader and binary.Read (reflection)
	cursor := 0

	// 1. Status Byte
	status := w.readBuf[cursor]
	cursor++

	if status == 1 { // Error from Python
		msgLen := binary.BigEndian.Uint32(w.readBuf[cursor:])
		cursor += 4

		if msgLen > 1<<20 { // Safety check: Max 1MB error message
			return nil, fmt.Errorf("python worker returned oversized error message: %d bytes", msgLen)
		}
		msg := w.readBuf[cursor : cursor+int(msgLen)]
		return nil, fmt.Errorf("python worker error: %s", string(msg))
	}

	// 2. Num Faces
	numFaces := binary.BigEndian.Uint32(w.readBuf[cursor:])
	cursor += 4

	results := make([]types.FaceResult, numFaces)

	// OPTIMIZATION: Allocate ONE contiguous backing array for all vectors in this frame.
	// CRITICAL FIX: We cannot reuse a persistent buffer (w.vecBuf) here because these slices
	// escape to the results channel and are read asynchronously by the aggregator.
	allVecs := make([]float64, int(numFaces)*512)

	for i := 0; i < int(numFaces); i++ {
		// [Box: 16B]
		x1 := int32(binary.BigEndian.Uint32(w.readBuf[cursor:]))
		y1 := int32(binary.BigEndian.Uint32(w.readBuf[cursor+4:]))
		x2 := int32(binary.BigEndian.Uint32(w.readBuf[cursor+8:]))
		y2 := int32(binary.BigEndian.Uint32(w.readBuf[cursor+12:]))
		cursor += 16

		// [Vec: 2048B]
		// Slice from our pre-allocated backing array
		vec64 := allVecs[i*512 : (i+1)*512]
		for j := 0; j < 512; j++ {
			bits := binary.BigEndian.Uint32(w.readBuf[cursor:])
			vec64[j] = float64(math.Float32frombits(bits))
			cursor += 4
		}

		// [Quality: 4B]
		qualityBits := binary.BigEndian.Uint32(w.readBuf[cursor:])
		quality := math.Float32frombits(qualityBits)
		cursor += 4

		// [ImgLen: 4B] + [ImgData]
		imgLen := binary.BigEndian.Uint32(w.readBuf[cursor:])
		cursor += 4

		// Copy image data to a new slice to prevent corruption when readBuf is reused
		imgData := make([]byte, imgLen)
		copy(imgData, w.readBuf[cursor:cursor+int(imgLen)])
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

// Close cleans up the worker resources, closing pipes and waiting for the process to exit.
func (w *PythonWorker) Close() {
	w.Stdin.Close()
	w.DataPipe.Close()
	w.Cmd.Wait()
}
