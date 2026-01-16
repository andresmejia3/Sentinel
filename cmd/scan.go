package cmd

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"runtime"
	"sync"

	"github.com/andresmejia3/sentinel/internal/types"
	"github.com/andresmejia3/sentinel/internal/utils"
	"github.com/andresmejia3/sentinel/internal/worker"
	"github.com/schollz/progressbar/v3"
	"github.com/spf13/cobra"
)

const megabyte = 1024 * 1024

// Flag variables
var (
	videoPath  string
	nthFrame   int
	numEngines int
)

var scanCmd = &cobra.Command{
	Use:   "scan",
	Short: "Scan video with parallel engines",
	Run: func(cmd *cobra.Command, args []string) {
		runScan()
	},
}

func init() {
	scanCmd.Flags().StringVarP(&videoPath, "input", "i", "", "Path to video")
	scanCmd.Flags().IntVarP(&nthFrame, "nth-frame", "n", 10, "AI keyframe interval (e.g. scan every 10th frame)")
	scanCmd.Flags().IntVarP(&numEngines, "engines", "e", runtime.NumCPU(), "Number of parallel engine workers")

	scanCmd.MarkFlagRequired("input")
	rootCmd.AddCommand(scanCmd)
}

// Buffer pool to reduce GC pressure during scanning
var frameBufferPool = sync.Pool{
	New: func() interface{} { return make([]byte, 0, megabyte) },
}

func runScan() {
	validateScanFlags()

	// 0. Get total frames for progress bar
	totalVideoFrames := utils.GetTotalFrames(videoPath)

	if totalVideoFrames <= 0 {
		// Fallback to a spinner or unknown total if ffprobe fails
		totalVideoFrames = -1
	}

	bar := progressbar.NewOptions(totalVideoFrames,
		progressbar.OptionSetDescription("🔍 Sentinel Scanning"),
		progressbar.OptionSetWriter(os.Stderr), // Write bar to Stderr
		progressbar.OptionShowCount(),
	)

	taskChan := make(chan types.FrameTask, numEngines*2)
	var wg sync.WaitGroup

	// 1. Spawn the Engine Pool
	for i := 0; i < numEngines; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			startWorker(workerID, taskChan)
		}(i)
	}

	// 2. Start FFmpeg
	ffmpeg := utils.NewFFmpegCmd(videoPath)

	var stderrBuf bytes.Buffer
	ffmpeg.Stderr = &stderrBuf

	ffmpegOut, err := ffmpeg.StdoutPipe()
	if err != nil {
		utils.Die("Failed to create FFmpeg stdout pipe", err, nil)
	}
	defer ffmpegOut.Close() // Ensure pipe is closed to prevent leaks/zombies

	if err := ffmpeg.Start(); err != nil {
		utils.Die("Failed to start FFmpeg", err, nil)
	}

	// 3. Frame Splitter & Nth-Frame Logic
	scanner := bufio.NewScanner(ffmpegOut)
	scanner.Buffer(make([]byte, megabyte), 64*megabyte)
	scanner.Split(utils.SplitJpeg)

	totalFrames := 0
	sentFrames := 0
	for scanner.Scan() {
		totalFrames++
		bar.Add(1) // Update progress bar for every frame read

		if totalFrames%nthFrame == 0 {
			// Get buffer from pool
			buf := frameBufferPool.Get().([]byte)
			if cap(buf) < len(scanner.Bytes()) {
				buf = make([]byte, len(scanner.Bytes()))
			}
			buf = buf[:len(scanner.Bytes())]
			copy(buf, scanner.Bytes())
			taskChan <- types.FrameTask{Index: totalFrames, Data: buf}
			sentFrames++
		}
	}

	// Check for scanner errors (e.g. token too long, unexpected EOF)
	if err := scanner.Err(); err != nil {
		utils.Die("Frame scanner failed", err, nil)
	}

	// 4. Cleanup & Completion Check
	if err := ffmpeg.Wait(); err != nil {
		if stderrBuf.Len() > 0 {
			fmt.Fprintf(os.Stderr, "\nFFmpeg Logs:\n%s\n", stderrBuf.String())
		}
		utils.Die("FFmpeg execution failed", err, nil)
	}

	close(taskChan)
	wg.Wait()

	bar.Finish()
	fmt.Fprintf(os.Stderr, "\n🏁 Scan Complete. Processed %d keyframes out of %d total.\n", sentFrames, totalFrames)
}

func startWorker(id int, tasks <-chan types.FrameTask) {
	worker, err := worker.NewPythonWorker(id)
	if err != nil {
		utils.Die("Worker startup failed", err, nil)
	}
	defer worker.Close()

	for task := range tasks {
		resp, err := worker.Communicate(task.Data)

		// Return buffer to pool immediately after sending
		frameBufferPool.Put(task.Data)

		if err != nil {
			// DRAIN: Wait for process to exit and capture final stderr logs
			worker.Close()
			utils.Die("Python crashed", err, worker.Cmd)
		}

		// Now we can actually USE the data!
		var faces []types.FaceResult
		if err := json.Unmarshal(resp, &faces); err != nil {
			// Check if it's a Python error object (e.g. {"error": "..."})
			var errorResult types.ErrorResult
			if json.Unmarshal(resp, &errorResult) == nil && errorResult.Error != "" {
				fmt.Fprintf(os.Stderr, "\n⚠️ Worker %d Logic Error: %s\n", id, errorResult.Error)
				continue
			}

			// Genuine unmarshal failure (garbage data)
			fmt.Fprintf(os.Stderr, "\n⚠️ Worker %d JSON Malformed: %v\n", id, err)
			continue
		}

		// TODO: Send 'faces' to a results channel for DB insertion
	}
}

func validateScanFlags() {
	info, err := os.Stat(videoPath)
	if err != nil {
		if os.IsNotExist(err) {
			utils.Die("Input file does not exist", err, nil)
		}
		utils.Die("Unable to access input file", err, nil)
	}
	if info.IsDir() {
		utils.Die("Input path is a directory, expected a video file", nil, nil)
	}
	if nthFrame < 1 {
		utils.Die("Invalid nth-frame interval", fmt.Errorf("must be >= 1, got %d", nthFrame), nil)
	}
	if numEngines < 1 {
		numEngines = 1
	}
}
