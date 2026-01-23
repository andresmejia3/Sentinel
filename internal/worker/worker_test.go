package worker

import (
	"bytes"
	"encoding/binary"
	"math"
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
	// Protocol: [Status:0] [NumFaces:1] [Box] [Vec] [Qual] [ImgLen] [Img]

	payload := new(bytes.Buffer)
	payload.WriteByte(0)                               // Status OK
	binary.Write(payload, binary.BigEndian, uint32(1)) // 1 Face

	// Face Data
	binary.Write(payload, binary.BigEndian, [4]int32{10, 10, 20, 20}) // Box

	vec := [512]float32{}
	vec[0] = 0.5                                 // Set one value to verify
	binary.Write(payload, binary.BigEndian, vec) // Vec

	binary.Write(payload, binary.BigEndian, float32(0.99)) // Quality

	imgData := []byte{0xCA, 0xFE}
	binary.Write(payload, binary.BigEndian, uint32(len(imgData))) // ImgLen
	payload.Write(imgData)                                        // ImgData

	fakePayload := payload.Bytes()

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
	if len(resp) != 1 {
		t.Fatalf("Expected 1 face, got %d", len(resp))
	}
	// Use epsilon for float comparison
	if math.Abs(resp[0].Vec[0]-0.5) > 1e-9 {
		t.Errorf("Expected vector[0] approx 0.5, got %f", resp[0].Vec[0])
	}
}

func TestProcessFrame_Error(t *testing.T) {
	// 1. Setup Mocks
	stdinMock := &MockCloser{Buffer: new(bytes.Buffer)}
	dataPipeMock := &MockCloser{Buffer: new(bytes.Buffer)}

	// 2. Pre-fill dataPipeMock with an ERROR response from "Python"
	// Protocol: [Status:1] [MsgLen] [Msg]

	payload := new(bytes.Buffer)
	payload.WriteByte(1) // Status ERROR

	errMsg := "Python Exception: Import Error"
	binary.Write(payload, binary.BigEndian, uint32(len(errMsg)))
	payload.WriteString(errMsg)

	fakePayload := payload.Bytes()

	// Write the length header
	binary.Write(dataPipeMock, binary.BigEndian, uint32(len(fakePayload)))
	dataPipeMock.Write(fakePayload)

	// 3. Create Worker
	w := &PythonWorker{
		ID:       1,
		Stdin:    stdinMock,
		DataPipe: dataPipeMock,
	}

	// 4. Execute
	_, err := w.ProcessFrame([]byte("frame"))

	// 5. Assertions
	if err == nil {
		t.Fatal("Expected error, got nil")
	}
	if err.Error() != "python worker error: "+errMsg {
		t.Errorf("Expected error message '%s', got '%v'", "python worker error: "+errMsg, err)
	}
}
