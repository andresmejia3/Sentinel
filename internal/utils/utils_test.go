package utils

import (
	"bufio"
	"bytes"
	"math"
	"os"
	"path/filepath"
	"slices"
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

func TestGenerateVideoIDDetectsMiddleChunkChanges(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "episode.mp4")

	size := 32 * videoIDChunkSize
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}

	block := bytes.Repeat([]byte{0xAB}, int(videoIDChunkSize))
	for written := int64(0); written < size; written += int64(len(block)) {
		if _, err := f.Write(block); err != nil {
			f.Close()
			t.Fatal(err)
		}
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	id1, err := GenerateVideoID(path)
	if err != nil {
		t.Fatalf("GenerateVideoID returned error: %v", err)
	}

	offsets := videoIDChunkOffsets(size, videoIDSampleCount(size))
	if len(offsets) < 3 {
		t.Fatalf("expected multiple chunk offsets, got %v", offsets)
	}

	middleOffset := offsets[len(offsets)/2] + 12345
	f, err = os.OpenFile(path, os.O_WRONLY, 0644)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteAt([]byte{0xCD}, middleOffset); err != nil {
		f.Close()
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	id2, err := GenerateVideoID(path)
	if err != nil {
		t.Fatalf("GenerateVideoID returned error after middle change: %v", err)
	}
	if id1 == id2 {
		t.Fatal("expected sampled middle-chunk change to alter video ID")
	}
}

func TestFmtTime(t *testing.T) {
	tests := []struct {
		seconds float64
		want    string
	}{
		{0, "00:00:00"},
		{65, "00:01:05"},
		{3661, "01:01:01"},
	}

	for _, tt := range tests {
		if got := FmtTime(tt.seconds); got != tt.want {
			t.Errorf("FmtTime(%v) = %v, want %v", tt.seconds, got, tt.want)
		}
	}
}

func TestParseFPSString(t *testing.T) {
	tests := []struct {
		name    string
		raw     string
		want    float64
		wantErr bool
	}{
		{name: "fraction", raw: "30000/1001", want: 30000.0 / 1001.0},
		{name: "integer", raw: "30", want: 30},
		{name: "not available", raw: "N/A", wantErr: true},
		{name: "zero denominator", raw: "0/0", wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseFPSString(tt.raw)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("expected parseFPSString(%q) to fail", tt.raw)
				}
				return
			}
			if err != nil {
				t.Fatalf("parseFPSString(%q) returned error: %v", tt.raw, err)
			}
			if math.Abs(got-tt.want) > 1e-9 {
				t.Fatalf("parseFPSString(%q) = %f, want %f", tt.raw, got, tt.want)
			}
		})
	}
}

func TestNewFFmpegEncoderTranscodesAudioForMP4Outputs(t *testing.T) {
	cmd := NewFFmpegEncoder(t.Context(), "input.webm", "output/redacted.mp4", 30, 1920, 1080)

	if !slices.Contains(cmd.Args, "aac") {
		t.Fatalf("expected mp4 output to transcode audio to AAC, args: %v", cmd.Args)
	}
	if slices.Contains(cmd.Args, "copy") {
		t.Fatalf("expected mp4 output to avoid stream-copying audio, args: %v", cmd.Args)
	}
}

func TestNewFFmpegEncoderCopiesAudioForNonMP4Outputs(t *testing.T) {
	cmd := NewFFmpegEncoder(t.Context(), "input.mov", "output/redacted.mkv", 30, 1920, 1080)

	if !slices.Contains(cmd.Args, "copy") {
		t.Fatalf("expected mkv output to keep audio copy path, args: %v", cmd.Args)
	}
	if slices.Contains(cmd.Args, "aac") {
		t.Fatalf("expected mkv output to avoid forced AAC transcode, args: %v", cmd.Args)
	}
}
