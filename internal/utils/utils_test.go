package utils

import (
	"bufio"
	"bytes"
	"os"
	"testing"
)

func TestSplitJpeg(t *testing.T) {
	// Construct a stream containing: [Garbage] [JPEG] [Garbage]
	// SOI (Start of Image): FF D8
	// EOI (End of Image):   FF D9

	jpegData := []byte{0xFF, 0xD8, 0x01, 0x02, 0x03, 0xFF, 0xD9}

	streamData := []byte{0x00, 0x00} // Garbage at start
	streamData = append(streamData, jpegData...)
	streamData = append(streamData, []byte{0x00, 0x00}...) // Garbage at end

	// Use bufio.Scanner with our custom Split function
	scanner := bufio.NewScanner(bytes.NewReader(streamData))
	scanner.Split(SplitJpeg)

	// Scan() should skip the first garbage bytes and find the JPEG
	if !scanner.Scan() {
		t.Fatal("Expected to find a token, got EOF")
	}

	// Verify the extracted token is exactly the JPEG
	if !bytes.Equal(scanner.Bytes(), jpegData) {
		t.Errorf("Expected %X, got %X", jpegData, scanner.Bytes())
	}

	// Scan() again should return false (EOF) because the trailing garbage is not a JPEG
	if scanner.Scan() {
		t.Error("Expected only one token, found more")
	}
}

func TestGenerateVideoID(t *testing.T) {
	// Integration test using the OS filesystem
	tmp, err := os.CreateTemp("", "video_test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmp.Name())

	// Write dummy content
	if _, err := tmp.Write([]byte("fake video content")); err != nil {
		t.Fatal(err)
	}
	tmp.Close()

	id, err := GenerateVideoID(tmp.Name())
	if err != nil || id == "" {
		t.Errorf("Failed to generate ID: %v", err)
	}

	// Verify Determinism
	id2, _ := GenerateVideoID(tmp.Name())
	if id != id2 {
		t.Errorf("Hash is not deterministic. Got %s, then %s", id, id2)
	}

	// Verify Sensitivity (Change content -> Change ID)
	f, _ := os.OpenFile(tmp.Name(), os.O_APPEND|os.O_WRONLY, 0644)
	f.Write([]byte(" modification"))
	f.Close()

	id3, _ := GenerateVideoID(tmp.Name())
	if id == id3 {
		t.Error("Hash did not change after file modification")
	}
}
