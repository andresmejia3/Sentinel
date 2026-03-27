package cmd

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/andresmejia3/sentinel/internal/utils"
	"github.com/andresmejia3/sentinel/internal/worker"
	"github.com/spf13/cobra"
)

var findOpts Options

var findCmd = &cobra.Command{
	Use:   "find <image_path>",
	Short: "Search for a face in the indexed video database",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		cmd.SilenceUsage = true
		return runFind(cmd.Context(), args[0], findOpts)
	},
}

func init() {
	findCmd.Flags().Float64VarP(&findOpts.MatchThreshold, "threshold", "t", 0.6, "Face matching threshold (lower is stricter)")
	findCmd.Flags().Float64VarP(&findOpts.DetectionThreshold, "detection-threshold", "D", 0.5, "Face detection confidence threshold")
	findCmd.Flags().BoolVarP(&findOpts.DebugScreenshots, "debug", "d", false, "Enable debug screenshots")
	findCmd.Flags().StringVar(&findOpts.WorkerTimeout, "worker-timeout", "30s", "Timeout for AI worker processing per frame")
	rootCmd.AddCommand(findCmd)
}

func runFind(ctx context.Context, imagePath string, opts Options) error {
	absPath, err := filepath.Abs(imagePath)
	if err != nil {
		return fmt.Errorf("failed to resolve absolute path for %s: %w", imagePath, err)
	}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	if err := validateFindFlags(absPath, opts); err != nil {
		return err
	}

	workerTimeout, err := time.ParseDuration(opts.WorkerTimeout)
	if err != nil {
		return fmt.Errorf("failed to parse worker-timeout '%s': %w", opts.WorkerTimeout, err)
	}

	fmt.Fprintln(os.Stderr, "🚀 Starting AI Engine...")
	cfg := worker.ScanConfig{
		Debug:              opts.DebugScreenshots,
		DetectionThreshold: opts.DetectionThreshold,
		ReadTimeout:        workerTimeout,
	}

	// We use ID 0 for this ad-hoc worker
	w, err := worker.NewPythonScanWorker(ctx, 0, cfg)
	if err != nil {
		return &utils.ContextualError{Context: "AI Worker Failed to Start", Err: err}
	}
	defer w.Close()

	imgData, err := os.ReadFile(absPath)
	if err != nil {
		return fmt.Errorf("failed to read image file at %s: %w", absPath, err)
	}

	fmt.Fprintln(os.Stderr, "🔍 Analyzing face...")
	faces, err := w.ProcessScanFrame(imgData)
	if err != nil {
		utils.ShowError("AI processing failed", err, w.Cmd)
		return &utils.SilentError{Err: err}
	}

	if len(faces) == 0 {
		fmt.Fprintln(os.Stderr, "❌ No faces detected in the provided image.")
		return nil
	}

	// Pick largest face if multiple
	bestFace := faces[0]

	getArea := func(loc []int) float64 {
		if len(loc) < 4 {
			return 0
		}
		return math.Abs(float64(loc[2]-loc[0]) * float64(loc[3]-loc[1]))
	}

	if len(faces) > 1 {
		fmt.Fprintf(os.Stderr, "⚠️  Multiple faces detected (%d). Using the largest face.\n", len(faces))

		maxArea := getArea(bestFace.Loc)
		for _, f := range faces[1:] {
			area := getArea(f.Loc)
			if area > maxArea {
				maxArea = area
				bestFace = f
			}
		}
	}

	// Ensure the chosen face has valid coordinates before proceeding
	if len(bestFace.Loc) < 4 {
		return fmt.Errorf("AI engine returned malformed face coordinates")
	}

	// Safety Check: Verify vector dimension of the chosen face before DB query
	if len(bestFace.Vec) != 512 {
		return fmt.Errorf("AI engine error: expected 512-dimensional vector, got %d", len(bestFace.Vec))
	}

	fmt.Fprintln(os.Stderr, "🗄️  Searching database...")
	variantID, identityID, identityName, variantName, err := DB.FindClosestIdentity(ctx, bestFace.Vec, opts.MatchThreshold)
	if err != nil {
		return fmt.Errorf("database search failed: %w", err)
	}

	if variantID == -1 {
		fmt.Fprintln(os.Stderr, "❌ No match found in database.")
		return nil
	}

	variantPart := ""
	if variantName != "Default" && variantName != "" {
		variantPart = fmt.Sprintf(" (%s)", variantName)
	}

	fmt.Printf("✅ Found Match: %s%s (ID: %d, Variant: %d)\n", identityName, variantPart, identityID, variantID)

	intervals, err := DB.GetIntervalsForIdentity(ctx, identityID)
	if err != nil {
		return fmt.Errorf("failed to retrieve history: %w", err)
	}

	if len(intervals) == 0 {
		fmt.Fprintln(os.Stderr, "ℹ️  No historical occurrences found in the database.")
		return nil
	}

	// Sort by Path for humans, then by ID to keep file versions together,
	// and finally by Start time for a chronological timeline.
	sort.Slice(intervals, func(i, j int) bool {
		if intervals[i].VideoPath != intervals[j].VideoPath {
			return intervals[i].VideoPath < intervals[j].VideoPath
		}
		if intervals[i].VideoID != intervals[j].VideoID {
			return intervals[i].VideoID < intervals[j].VideoID
		}
		return intervals[i].Start < intervals[j].Start
	})

	fmt.Println("") // Spacing
	var lastVideoID string
	for i, inv := range intervals {
		// Content Hash (VideoID) is the definitive grouping key.
		// Even if paths are the same, different IDs mean different videos.
		if i == 0 || inv.VideoID != lastVideoID {
			if i > 0 {
				fmt.Println("")
			}

			// If the VideoID is different but the path is the same as the last one,
			// we append a short hash to alert the user that these are different files.
			header := inv.VideoPath
			if i > 0 && inv.VideoPath == intervals[i-1].VideoPath {
				header = fmt.Sprintf("%s [hash: %s]", inv.VideoPath, inv.VideoID[:8])
			}
			fmt.Printf("🎬 %s\n", header)
			lastVideoID = inv.VideoID
		}

		duration := inv.End - inv.Start
		fmt.Printf("   👉 %s - %s (%.1fs)\n", utils.FmtTime(inv.Start), utils.FmtTime(inv.End), duration)
	}

	return nil
}

func validateFindFlags(imagePath string, opts Options) error {
	info, err := os.Stat(imagePath)
	if err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("input file does not exist: %w", err)
		}
		return fmt.Errorf("unable to access input file: %w", err)
	}
	if info.IsDir() {
		return fmt.Errorf("input path is a directory, not an image file")
	}
	if opts.MatchThreshold < 0 || opts.MatchThreshold > 1.0 {
		return fmt.Errorf("invalid match threshold: must be between 0.0 and 1.0, got %f", opts.MatchThreshold)
	}
	if opts.DetectionThreshold < 0 || opts.DetectionThreshold > 1.0 {
		return fmt.Errorf("invalid detection threshold: must be between 0.0 and 1.0, got %f", opts.DetectionThreshold)
	}
	if _, err := time.ParseDuration(opts.WorkerTimeout); err != nil {
		return fmt.Errorf("invalid worker-timeout format: %w", err)
	}
	return nil
}
