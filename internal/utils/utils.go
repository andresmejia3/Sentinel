package utils

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

// --- 1. Process Safety & Command Wrapping ---

// SafeCommand wraps a standard exec.Cmd with a buffer to catch Stderr (Python logs)
// This ensures we don't lose critical crash information if a worker dies.
type SafeCommand struct {
	*exec.Cmd
	Stderr *bytes.Buffer
}

// NewSafeCommand initializes a command and attaches a buffer to its Stderr pipe
// It prepares the command for execution but does not start it.
func NewSafeCommand(name string, args ...string) *SafeCommand {
	cmd := exec.Command(name, args...)
	stderr := &bytes.Buffer{}
	cmd.Stderr = stderr // Capture stderr for crash logs, but don't print to terminal
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

	if s != nil && s.ProcessState != nil {
		fmt.Fprintf(os.Stderr, "EXIT CODE: %d (%s)\n", s.ProcessState.ExitCode(), s.ProcessState.String())

		// Detect OOM Kill (Exit Code -1 + "killed" on Unix, or 137)
		exitCode := s.ProcessState.ExitCode()
		status := strings.ToLower(s.ProcessState.String())
		if (exitCode == -1 && strings.Contains(status, "killed")) || exitCode == 137 {
			fmt.Fprintf(os.Stderr, "\n📉 MEMORY ERROR: The worker was killed by the OS (OOM).\n")
			fmt.Fprintf(os.Stderr, "   Your Docker container ran out of RAM spawning multiple AI models.\n")
			fmt.Fprintf(os.Stderr, "   👉 SOLUTION: Run with fewer engines using the '-e' flag.\n")
			fmt.Fprintf(os.Stderr, "      Example: ./sentinel-docker.sh scan -i ... -e 1\n")
		}
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
// It returns 0 if the count fails, allowing the scanner to fallback to a spinner.
func GetTotalFrames(path string) int {
	// 0. Check dependency
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

	// 1. Fast Path: Check Container Metadata
	// This is instant but might return "N/A" or be inaccurate for VFR.
	cmdFast := exec.Command("ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=nb_frames", "-of", "json", path)
	if out, err := cmdFast.Output(); err == nil {
		var res ffprobeOutput
		if json.Unmarshal(out, &res) == nil && len(res.Streams) > 0 {
			if count, err := strconv.Atoi(res.Streams[0].NbFrames); err == nil && count > 0 {
				return count
			}
		}
	}

	// 2. Slow Path: Count Packets (Fallback)
	fmt.Fprintf(os.Stderr, "⏳ Metadata missing. Counting frames (this may take a moment)...\n")
	cmd := exec.Command("ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
		"-show_entries", "stream=nb_read_packets", "-of", "json", path)

	cmd.Stderr = os.Stderr
	out, err := cmd.Output()

	if err != nil {
		fmt.Fprintf(os.Stderr, "ffprobe failed: %v\n", err)
		return 0
	}

	var res ffprobeOutput
	if err := json.Unmarshal(out, &res); err != nil {
		fmt.Fprintf(os.Stderr, "ffprobe JSON parse error: %v\n", err)
		return 0
	}
	if len(res.Streams) == 0 {
		return 0
	}

	count, err := strconv.Atoi(res.Streams[0].NbReadPackets)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ffprobe integer parse error: %v\n", err)
		return 0
	}
	return count
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
func NewFFmpegCmd(inputPath string) *exec.Cmd {
	// Using -vcodec mjpeg ensures we get JPEGs Go can split
	// Added -hide_banner and -loglevel error to prevent memory bloat in stderr buffer
	return exec.Command("ffmpeg", "-hide_banner", "-loglevel", "error", "-i", inputPath, "-f", "image2pipe", "-vcodec", "mjpeg", "-")
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
	fmt.Fprintf(hash, "%s|%d|", filepath.Base(path), info.Size())
	hash.Write(headBuf[:nHead])
	hash.Write(tailBuf[:nTail])

	return hex.EncodeToString(hash.Sum(nil)), nil
}

// GetVideoFPS returns the average frame rate of the video.
func GetVideoFPS(path string) (float64, error) {
	if _, err := exec.LookPath("ffprobe"); err != nil {
		fmt.Fprintf(os.Stderr, "⚠️  ffprobe not found. It is required for processing.\n")
		return 0, err
	}
	cmd := exec.Command("ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1", path)
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
