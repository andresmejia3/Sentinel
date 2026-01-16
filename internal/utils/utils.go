package utils

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
)

// --- 1. Process Safety & Command Wrapping ---

// SafeCommand wraps a standard exec.Cmd with a buffer to catch Stderr (Python logs)
type SafeCommand struct {
	*exec.Cmd
	Stderr *bytes.Buffer
}

// NewSafeCommand initializes a command and attaches a buffer to its Stderr pipe
func NewSafeCommand(name string, args ...string) *SafeCommand {
	cmd := exec.Command(name, args...)
	stderr := &bytes.Buffer{}
	cmd.Stderr = stderr
	return &SafeCommand{Cmd: cmd, Stderr: stderr}
}

// Die is the unified exit strategy for Sentinel.
// It prints a formatted error box and dumps Python logs if a SafeCommand is provided.
func Die(context string, err error, s *SafeCommand) {
	fmt.Fprintf(os.Stderr, "\n---------------------------------------------------------\n")
	fmt.Fprintf(os.Stderr, "🚨 SENTINEL ERROR: %s\n", context)
	if err != nil {
		fmt.Fprintf(os.Stderr, "DETAILS: %v\n", err)
	}

	// If we have a SafeCommand and it captured logs, print them.
	if s != nil && s.Stderr.Len() > 0 {
		fmt.Fprintf(os.Stderr, "\nPYTHON CRASH LOGS:\n%s\n", s.Stderr.String())
	}
	fmt.Fprintf(os.Stderr, "---------------------------------------------------------\n")
	os.Exit(1)
}

// --- 2. Video Engine (Shared by Scan & Redact) ---

var (
	JpegSOI = []byte{0xFF, 0xD8} // Start of Image
	JpegEOI = []byte{0xFF, 0xD9} // End of Image
)

// GetTotalFrames uses ffprobe to count packets for the progress bar
func GetTotalFrames(path string) int {
	cmd := exec.Command("ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
		"-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", path)

	cmd.Stderr = os.Stderr
	out, err := cmd.Output()

	if err != nil {
		fmt.Fprintf(os.Stderr, "ffprobe failed: %v\n", err)
		return 0
	}

	cleanOut := strings.TrimRight(strings.TrimSpace(string(out)), ",")
	count, err := strconv.Atoi(cleanOut)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ffprobe parse error: %v\n", err)
		return 0
	}
	return count
}

// SplitJpeg is the custom splitter for bufio.Scanner
func SplitJpeg(data []byte, atEOF bool) (advance int, token []byte, err error) {
	if atEOF && len(data) == 0 {
		return 0, nil, nil
	}
	start := bytes.Index(data, JpegSOI)
	if start == -1 {
		return 0, nil, nil
	}
	end := bytes.Index(data[start:], JpegEOI)
	if end == -1 {
		return 0, nil, nil
	}
	return start + end + 2, data[start : start+end+2], nil
}

// NewFFmpegCmd creates a standard decoder pipe
func NewFFmpegCmd(inputPath string) *exec.Cmd {
	// Using -vcodec mjpeg ensures we get JPEGs Go can split
	// Added -hide_banner and -loglevel error to prevent memory bloat in stderr buffer
	return exec.Command("ffmpeg", "-hide_banner", "-loglevel", "error", "-i", inputPath, "-f", "image2pipe", "-vcodec", "mjpeg", "-")
}
