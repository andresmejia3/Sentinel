package cmd

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"text/tabwriter"
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
	findCmd.Flags().Float64VarP(&findOpts.MatchThreshold, "threshold", "t", 0.6, "Face matching threshold")
	findCmd.Flags().Float64VarP(&findOpts.DetectionThreshold, "detection-threshold", "D", 0.5, "Face detection confidence threshold")
	findCmd.Flags().BoolVarP(&findOpts.DebugScreenshots, "debug", "d", false, "Enable debug screenshots")
	rootCmd.AddCommand(findCmd)
}

func runFind(ctx context.Context, imagePath string, opts Options) error {
	if _, err := os.Stat(imagePath); os.IsNotExist(err) {
		utils.ShowError("Input file does not exist", err, nil)
		return err
	}

	fmt.Fprintln(os.Stderr, "🚀 Starting AI Engine...")
	cfg := worker.ScanConfig{
		Debug:              opts.DebugScreenshots,
		DetectionThreshold: opts.DetectionThreshold,
		ReadTimeout:        60 * time.Second,
	}

	// We use ID 0 for this ad-hoc worker
	w, err := worker.NewPythonScanWorker(ctx, 0, cfg)
	if err != nil {
		utils.ShowError("Failed to start AI worker", err, nil)
		return err
	}
	defer w.Close()

	imgData, err := os.ReadFile(imagePath)
	if err != nil {
		utils.ShowError("Failed to read image file", err, nil)
		return err
	}

	fmt.Fprintln(os.Stderr, "🔍 Analyzing face...")
	faces, err := w.ProcessScanFrame(imgData)
	if err != nil {
		utils.ShowError("AI processing failed", err, w.Cmd)
		return err
	}

	if len(faces) == 0 {
		fmt.Println("❌ No faces detected in the provided image.")
		return nil
	}

	// Pick largest face if multiple
	bestFace := faces[0]
	if len(faces) > 1 {
		fmt.Printf("⚠️  Multiple faces detected (%d). Using the largest face.\n", len(faces))
		maxArea := (bestFace.Loc[2] - bestFace.Loc[0]) * (bestFace.Loc[3] - bestFace.Loc[1])
		for _, f := range faces[1:] {
			area := (f.Loc[2] - f.Loc[0]) * (f.Loc[3] - f.Loc[1])
			if area > maxArea {
				maxArea = area
				bestFace = f
			}
		}
	}

	fmt.Fprintln(os.Stderr, "🗄️  Searching database...")
	id, name, err := DB.FindClosestIdentity(ctx, bestFace.Vec, opts.MatchThreshold)
	if err != nil {
		utils.ShowError("Database search failed", err, nil)
		return err
	}

	if id == -1 {
		fmt.Println("❌ No match found in database.")
		return nil
	}

	if name == "" {
		name = fmt.Sprintf("Identity %d", id)
	}
	fmt.Printf("✅ Found Match: %s (ID: %d)\n", name, id)

	intervals, err := DB.GetIdentityIntervals(ctx, id)
	if err != nil {
		utils.ShowError("Failed to retrieve history", err, nil)
		return err
	}

	if len(intervals) == 0 {
		fmt.Println("No recorded intervals found.")
		return nil
	}

	wOut := tabwriter.NewWriter(os.Stdout, 0, 0, 3, ' ', 0)
	fmt.Fprintln(wOut, "\nVIDEO\tTIME RANGE\tDURATION")
	fmt.Fprintln(wOut, "-----\t----------\t--------")

	for _, inv := range intervals {
		duration := inv.End - inv.Start
		fmt.Fprintf(wOut, "%s\t%s - %s\t%.1fs\n",
			filepath.Base(inv.VideoPath),
			fmtTime(inv.Start),
			fmtTime(inv.End),
			duration,
		)
	}
	wOut.Flush()

	return nil
}
