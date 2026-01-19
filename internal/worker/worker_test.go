package worker

import (
	"bytes"
	"encoding/binary"
	"testing"
)

// MockCloser wraps a bytes.Buffer to satisfy io.ReadCloser and io.WriteCloser interfaces.
// This allows us to use in-memory buffers as if they were OS Pipes.
type MockCloser struct {
	*bytes.Buffer
}

func (m *MockCloser) Close() error { return nil }

func TestProcessFrame(t *testing.T) {
	// 1. Setup Mocks
	// stdinMock simulates the pipe TO Python (we write to it)
	stdinMock := &MockCloser{Buffer: new(bytes.Buffer)}

	// dataPipeMock simulates the pipe FROM Python (we read from it)
	dataPipeMock := &MockCloser{Buffer: new(bytes.Buffer)}

	// 2. Pre-fill dataPipeMock with a fake response from "Python"
	// We simulate the protocol: [4-byte Length Header] + [JSON Payload]
	fakePayload := []byte(`[{"loc": [10,10,20,20], "vec": [0.1, 0.2]}]`)

	// Write the length header (Big Endian uint32)
	binary.Write(dataPipeMock, binary.BigEndian, uint32(len(fakePayload)))
	// Write the body
	dataPipeMock.Write(fakePayload)

	// 3. Create Worker with mocks injected
	w := &PythonWorker{
		ID:       1,
		Stdin:    stdinMock,
		DataPipe: dataPipeMock,
		// Cmd is nil because we aren't testing process management, just the protocol
	}

	// 4. Execute the function under test
	inputFrame := []byte{0xDE, 0xAD, 0xBE, 0xEF} // Fake image bytes
	resp, err := w.ProcessFrame(inputFrame)
	if err != nil {
		t.Fatalf("ProcessFrame failed: %v", err)
	}

	// 5. Assertions

	// Verify Go sent the correct data TO Python
	sentData := stdinMock.Bytes()
	// Expect 4 bytes header + 4 bytes data
	if len(sentData) != 4+len(inputFrame) {
		t.Errorf("Expected %d bytes sent, got %d", 4+len(inputFrame), len(sentData))
	}

	// Verify Go read the correct data FROM Python
	if !bytes.Equal(resp, fakePayload) {
		t.Errorf("Expected response %s, got %s", fakePayload, resp)
	}
}
