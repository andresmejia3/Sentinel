package utils

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

// --- Process Safety & Command Wrapping ---

// SafeCommand wraps a standard exec.Cmd with a buffer to catch Stderr (Python logs)
// This ensures we don't lose critical crash information if a worker dies.
type SafeCommand struct {
	*exec.Cmd
	Stderr *bytes.Buffer
}

// NewSafeCommand initializes a command and attaches a buffer to its Stderr pipe
// It prepares the command for execution but does not start it.
func NewSafeCommand(ctx context.Context, name string, args ...string) *SafeCommand {
	cmd := exec.CommandContext(ctx, name, args...)
	stderr := &bytes.Buffer{}
	cmd.Stdout = os.Stdout // Forward stdout for live logging (if Python prints anything)
	// The caller is responsible for assigning cmd.Stderr.
	return &SafeCommand{Cmd: cmd, Stderr: stderr}
}

// ShowError prints the formatted error box but DOES NOT exit the program.
// Use this when you want to return an error to main() for graceful shutdown.
func ShowError(context string, err error, s *SafeCommand) {
	fmt.Fprintf(os.Stderr, "\n---------------------------------------------------------\n")
	fmt.Fprintf(os.Stderr, "🚨 SENTINEL ERROR: %s\n", context)
	if err != nil {
		fmt.Fprintf(os.Stderr, "DETAILS: %v\n", err)
	}

	if s != nil && s.ProcessState != nil {
		fmt.Fprintf(os.Stderr, "EXIT CODE: %d (%s)\n", s.ProcessState.ExitCode(), s.ProcessState.String())

		// Detect OOM Kill (Exit Code -1 + "killed" on Unix, or 137)
		exitCode := s.ProcessState.ExitCode()
		status := strings.ToLower(s.ProcessState.String())
		if (exitCode == -1 && strings.Contains(status, "killed")) || exitCode == 137 {
			fmt.Fprintf(os.Stderr, "\n📉 MEMORY ERROR: The worker was killed by the OS (OOM).\n")
			fmt.Fprintf(os.Stderr, "   Your Docker container ran out of RAM spawning multiple AI models.\n")
			fmt.Fprintf(os.Stderr, "   👉 SOLUTION: Run with fewer engines using the '-e' flag.\n")
			fmt.Fprintf(os.Stderr, "      Example: ./sentinel scan -i ... -e 1\n")
		}
	}

	// If we have a SafeCommand and it captured logs, print them.
	if s != nil && s.Stderr.Len() > 0 {
		fmt.Fprintf(os.Stderr, "\nPYTHON CRASH LOGS:\n%s\n", s.Stderr.String())
	}
	fmt.Fprintf(os.Stderr, "---------------------------------------------------------\n")
}

// --- Video Engine (Shared by Scan & Redact) ---

var (
	JpegSOI = []byte{0xFF, 0xD8} // Start of Image
	JpegEOI = []byte{0xFF, 0xD9} // End of Image
)

// GetTotalFrames uses ffprobe to count packets for the progress bar
// It returns 0 if the count fails, allowing the scanner to fallback to a spinner.
func GetTotalFrames(ctx context.Context, path string) int {
	if _, err := exec.LookPath("ffprobe"); err != nil {
		fmt.Fprintf(os.Stderr, "⚠️  ffprobe not found. Cannot provide a progress bar estimation because of this.\n")
		return 0
	}

	// Helper struct for structured JSON parsing
	type ffprobeOutput struct {
		Streams []struct {
			NbFrames      string `json:"nb_frames"`
			NbReadPackets string `json:"nb_read_packets"`
		} `json:"streams"`
	}

	// Fast Path: Check Container Metadata
	// This is instant but might return "N/A" or be inaccurate for VFR.
	cmdFast := exec.CommandContext(ctx, "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=nb_frames", "-of", "json", path)
	if out, err := cmdFast.Output(); err == nil {
		var res ffprobeOutput
		if json.Unmarshal(out, &res) == nil && len(res.Streams) > 0 {
			if count, err := strconv.Atoi(res.Streams[0].NbFrames); err == nil && count > 0 {
				return count
			}
		}
	}

	// Optimization: Skip the slow packet count.
	// Reading the entire file to count packets delays startup significantly for large videos.
	// We return 0 to signal "Unknown Total", which causes the progress bar to use a spinner.
	fmt.Fprintf(os.Stderr, "⚠️  Video metadata missing frame count. Defaulting to spinner...\n")
	return 0
}

// SplitJpeg is the custom splitter for bufio.Scanner
// It locates the Start Of Image (FFD8) and End Of Image (FFD9) markers to extract full JPEG frames.
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
// It configures FFmpeg to output raw MJPEG frames to Stdout for ingestion.
func NewFFmpegCmd(ctx context.Context, inputPath string, nthFrame int) *exec.Cmd {
	// Using -vcodec mjpeg ensures we get JPEGs Go can split
	// Added -hide_banner and -loglevel error to prevent memory bloat in stderr buffer
	args := []string{"-hide_banner", "-loglevel", "error", "-i", inputPath}

	if nthFrame > 1 {
		// Optimization: Drop frames in FFmpeg before encoding to MJPEG
		args = append(args, "-vf", fmt.Sprintf("select=not(mod(n\\,%d))", nthFrame), "-vsync", "0")
	}

	args = append(args, "-f", "image2pipe", "-vcodec", "mjpeg", "-")
	return exec.CommandContext(ctx, "ffmpeg", args...)
}

// NewFFmpegRawDecoder creates a command to output raw RGBA frames to Stdout.
func NewFFmpegRawDecoder(ctx context.Context, inputPath string) *exec.Cmd {
	return exec.CommandContext(ctx, "ffmpeg",
		"-hide_banner", "-loglevel", "error",
		"-i", inputPath,
		"-f", "image2pipe",
		"-pix_fmt", "rgba",
		"-vcodec", "rawvideo", "-")
}

// NewFFmpegEncoder creates a command to encode raw JPEG frames from Stdin into a video file.
func NewFFmpegEncoder(ctx context.Context, outputPath string, fps float64, width, height int) *exec.Cmd {
	return exec.CommandContext(ctx, "ffmpeg",
		"-y", // Overwrite output
		"-f", "rawvideo",
		"-pix_fmt", "rgba",
		"-s", fmt.Sprintf("%dx%d", width, height),
		"-r", fmt.Sprintf("%f", fps),
		"-i", "-", // Read from Stdin
		"-c:v", "libx264",
		"-preset", "fast",
		"-crf", "23",
		"-pix_fmt", "yuv420p",
		outputPath,
	)
}

// GenerateVideoID creates a deterministic hash for the video file
// based on its path, size, and modification time.
func GenerateVideoID(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()

	info, err := f.Stat()
	if err != nil {
		return "", err
	}

	// Read first 32KB (Header)
	headBuf := make([]byte, 32*1024)
	nHead, err := f.Read(headBuf)
	if err != nil && err != io.EOF {
		return "", err
	}

	// Read last 32KB (Footer/Metadata)
	// Useful because some formats put critical unique info at the end
	tailBuf := make([]byte, 32*1024)
	var nTail int
	if info.Size() > 64*1024 {
		if _, err := f.Seek(-32*1024, io.SeekEnd); err == nil {
			nTail, _ = f.Read(tailBuf)
		}
	}

	// Hash: Filename + Size + Header + Footer
	// We use filepath.Base to allow moving the file without changing ID
	hash := sha256.New()
	fmt.Fprintf(hash, "%s|%d|", strings.ToLower(filepath.Base(path)), info.Size())
	hash.Write(headBuf[:nHead])
	hash.Write(tailBuf[:nTail])

	return hex.EncodeToString(hash.Sum(nil)), nil
}

// GetVideoFPS returns the average frame rate of the video.
func GetVideoFPS(ctx context.Context, path string) (float64, error) {
	if _, err := exec.LookPath("ffprobe"); err != nil {
		fmt.Fprintf(os.Stderr, "⚠️  ffprobe not found. It is required for processing.\n")
		return 0, err
	}
	cmd := exec.CommandContext(ctx, "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1", path)
	out, err := cmd.Output()
	if err != nil {
		return 0, err
	}

	// Output is typically "30/1" or "30000/1001"
	parts := strings.Split(strings.TrimSpace(string(out)), "/")
	if len(parts) == 1 {
		return strconv.ParseFloat(parts[0], 64)
	} else if len(parts) == 2 {
		num, err1 := strconv.ParseFloat(parts[0], 64)
		den, err2 := strconv.ParseFloat(parts[1], 64)
		if err1 != nil || err2 != nil || den == 0 {
			return 0, fmt.Errorf("invalid framerate format: %s", string(out))
		}
		return num / den, nil
	}
	return 0, fmt.Errorf("unknown framerate format: %s", string(out))
}

// GetVideoDimensions returns the width and height of the video stream.
func GetVideoDimensions(ctx context.Context, path string) (int, int, error) {
	if _, err := exec.LookPath("ffprobe"); err != nil {
		return 0, 0, err
	}
	// ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 input.mp4
	cmd := exec.CommandContext(ctx, "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", path)
	out, err := cmd.Output()
	if err != nil {
		return 0, 0, err
	}
	parts := strings.Split(strings.TrimSpace(string(out)), "x")
	if len(parts) != 2 {
		return 0, 0, fmt.Errorf("invalid dimension format: %s", string(out))
	}
	w, err1 := strconv.Atoi(parts[0])
	h, err2 := strconv.Atoi(parts[1])
	if err1 != nil || err2 != nil {
		return 0, 0, fmt.Errorf("invalid dimensions: %s", string(out))
	}
	return w, h, nil
}

// CosineDist calculates the cosine distance between two vectors.
// It returns a value between 0.0 (identical) and 2.0 (opposite).
// A result of 1.0 means the vectors are orthogonal.
func CosineDist(a, b []float64) float64 {
	// BCE (Bounds Check Elimination) Hint:
	// Proves to the compiler that a and b are large enough, removing checks inside the loop.
	if len(a) != len(b) || len(a) == 0 {
		return 1.0 // Return neutral distance for invalid input
	}
	_ = a[len(a)-1] // BCE
	_ = b[len(b)-1] // BCE

	var dot, sumA, sumB float64
	for i := range a {
		dot += a[i] * b[i]
		sumA += a[i] * a[i]
		sumB += b[i] * b[i]
	}

	if sumA == 0 || sumB == 0 {
		return 1.0
	}
	return 1.0 - (dot / (math.Sqrt(sumA) * math.Sqrt(sumB)))
}

// GetProcessRSS returns the Resident Set Size (RSS) in bytes for a given PID.
// Supported on Linux (via /proc). Returns 0 on other OSes or errors.
func GetProcessRSS(pid int) uint64 {
	path := fmt.Sprintf("/proc/%d/statm", pid)
	data, err := os.ReadFile(path)
	if err != nil {
		return 0 // Not supported or process gone
	}
	fields := strings.Fields(string(data))
	if len(fields) < 2 {
		return 0
	}
	// Second field is RSS in pages
	rssPages, _ := strconv.ParseUint(fields[1], 10, 64)
	return rssPages * uint64(os.Getpagesize())
}
