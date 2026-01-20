package worker

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"

	"github.com/andresmejia3/sentinel/internal/utils" // Using the SafeCommand wrapper
)

// PythonWorker manages the external Python process for neural inference.
type PythonWorker struct {
	ID       int
	Cmd      *utils.SafeCommand
	Stdin    io.WriteCloser
	DataPipe io.ReadCloser
}

// NewPythonWorker spawns a new Python process and sets up the IPC pipes (Stdin + Side-channel).
func NewPythonWorker(id int, debug bool) (*PythonWorker, error) {
	args := []string{"-u", "python/worker.py"}
	if debug {
		args = append(args, "--debug")
	}
	// 1. Initialize the SafeCommand we built
	py := utils.NewSafeCommand("python3", args...)

	// Create a side-channel pipe (FD 3) for clean data transfer
	r, w, err := os.Pipe()
	if err != nil {
		return nil, fmt.Errorf("failed to create pipe: %w", err)
	}
	// Pass the write-end to the child process. It will appear as FD 3.
	py.Cmd.ExtraFiles = []*os.File{w}

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

	return &PythonWorker{
		ID:       id,
		Cmd:      py,
		Stdin:    stdin,
		DataPipe: r,
	}, nil
}

// ProcessFrame sends a frame to the worker via Stdin and waits for the JSON response via the DataPipe.
// It handles the binary protocol: [4-byte Length][Payload].
func (w *PythonWorker) ProcessFrame(data []byte) ([]byte, error) {
	// Protocol: [Length][Data]
	if err := binary.Write(w.Stdin, binary.BigEndian, uint32(len(data))); err != nil {
		return nil, err
	}
	if _, err := w.Stdin.Write(data); err != nil {
		return nil, err
	}

	// Read Result
	// Now we read from our clean DataPipe, so no Magic Byte is needed.
	header := make([]byte, 4)
	if _, err := io.ReadFull(w.DataPipe, header); err != nil {
		return nil, err // This is where we catch the "ModuleNotFoundError" crash
	}

	respLen := binary.BigEndian.Uint32(header)
	respBody := make([]byte, respLen)
	_, err := io.ReadFull(w.DataPipe, respBody)
	return respBody, err
}

// Close cleans up the worker resources, closing pipes and waiting for the process to exit.
func (w *PythonWorker) Close() {
	w.Stdin.Close()
	w.DataPipe.Close()
	w.Cmd.Wait()
}
