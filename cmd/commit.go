package cmd

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/andresmejia3/sentinel/internal/store"
	"github.com/google/uuid"
	"github.com/spf13/cobra"
	"gopkg.in/yaml.v3"
)

var commitCmd = &cobra.Command{
	Use:   "commit <review.yaml>",
	Short: "Commit a reviewed scan file to the database with transaction ledger",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		cmd.SilenceUsage = true
		return runCommit(cmd.Context(), args[0])
	},
}

func init() {
	rootCmd.AddCommand(commitCmd)
}

func runCommit(ctx context.Context, stagingPath string) error {
	var data []byte
	data, err := os.ReadFile(stagingPath)
	if err != nil {
		return fmt.Errorf("failed to read review file: %w", err)
	}

	review, legacyFormat, err := readReviewDocument(data)
	if err != nil {
		return err
	}
	fmt.Println("🔍 Validating review file...")
	actions, intervals, skipCount, err := prepareCommitBatch(review, legacyFormat)
	if err != nil {
		return err
	}

	commitID := uuid.New().String()
	fmt.Printf("🚀 Starting Commit Batch: %s\n", commitID)

	if err := DB.ApplyCommitBatch(ctx, commitID, actions, review.VideoID, review.InputPath, intervals); err != nil {
		return fmt.Errorf("failed to apply commit batch: %w", err)
	}

	fmt.Println("---------------------------------------------------------")
	fmt.Printf("✅ Commit Batch Complete\n")
	fmt.Printf("🆔 Commit ID: %s  <-- Save this for rollback!\n", commitID)
	fmt.Printf("📊 Processed: %d  |  Skipped: %d\n", len(actions), skipCount)
	fmt.Println("---------------------------------------------------------")

	return nil
}

func prepareCommitBatch(review ReviewDocument, legacyFormat bool) ([]store.CommitAction, []store.CommitInterval, int, error) {
	if !legacyFormat {
		if review.VideoID == "" {
			return nil, nil, 0, fmt.Errorf("review file is missing video_id")
		}
		if review.InputPath == "" {
			return nil, nil, 0, fmt.Errorf("review file is missing input_path")
		}
	}

	var actions []store.CommitAction
	var intervals []store.CommitInterval
	skipCount := 0
	var unresolvedTracks []string

	for i, item := range review.Tracks {
		if item.Action == "" {
			trackID := item.TrackID
			if trackID == "" {
				trackID = fmt.Sprintf("item_%d", i)
			}
			unresolvedTracks = append(unresolvedTracks, trackID)
			continue
		}
		if item.Action == "discard" {
			skipCount++
			continue
		}
		if len(item.InternalVector) != 512 {
			return nil, nil, 0, fmt.Errorf("validation failed on item %d (%s): vector has %d dimensions, expected 512", i, item.TrackID, len(item.InternalVector))
		}
		if item.Action != "merge" && item.Action != "new_identity" && item.Action != "new_variant" {
			return nil, nil, 0, fmt.Errorf("validation failed on item %d (%s): invalid action '%s'", i, item.TrackID, item.Action)
		}
		if item.InternalCount <= 0 {
			return nil, nil, 0, fmt.Errorf("validation failed on item %d (%s): internal_count must be positive, got %d", i, item.TrackID, item.InternalCount)
		}
		if review.VideoID != "" {
			if item.StartTime < 0 || item.EndTime < 0 {
				return nil, nil, 0, fmt.Errorf("validation failed on item %d (%s): interval times must be non-negative", i, item.TrackID)
			}
			if item.EndTime < item.StartTime {
				return nil, nil, 0, fmt.Errorf("validation failed on item %d (%s): end_time must be >= start_time", i, item.TrackID)
			}
		}

		actions = append(actions, store.CommitAction{
			TrackID:      item.TrackID,
			Action:       item.Action,
			IdentityName: item.SuggestedIdentity,
			VariantName:  item.SuggestedVariant,
			Vector:       item.InternalVector,
			Count:        item.InternalCount,
		})
		if review.VideoID != "" {
			intervals = append(intervals, store.CommitInterval{
				TrackID:   item.TrackID,
				Start:     item.StartTime,
				End:       item.EndTime,
				FaceCount: item.InternalCount,
			})
		}
	}

	if len(unresolvedTracks) > 0 {
		return nil, nil, 0, fmt.Errorf("review file has unresolved tracks with blank action: %s", strings.Join(unresolvedTracks, ", "))
	}

	return actions, intervals, skipCount, nil
}

func readReviewDocument(data []byte) (ReviewDocument, bool, error) {
	var review ReviewDocument
	if err := yaml.Unmarshal(data, &review); err == nil {
		if review.Tracks != nil || review.VideoID != "" || review.InputPath != "" {
			return review, false, nil
		}
	}

	var legacy []StagingItem
	if err := yaml.Unmarshal(data, &legacy); err != nil {
		return ReviewDocument{}, false, fmt.Errorf("failed to parse review YAML: %w", err)
	}

	return ReviewDocument{Tracks: legacy}, true, nil
}
