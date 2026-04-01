package cmd

import (
	"context"
	"fmt"
	"os"
	"sort"
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
	if !legacyFormat {
		if review.VideoID == "" {
			return fmt.Errorf("review file is missing video_id")
		}
		if review.InputPath == "" {
			return fmt.Errorf("review file is missing input_path")
		}
	}
	review, err = hydrateReviewDocument(stagingPath, review)
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
			trackID := stagingItemLabel(item)
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
		trackKey := stagingItemKey(item)
		trackLabel := stagingItemLabel(item)
		if trackLabel == "" {
			trackLabel = fmt.Sprintf("item_%d", i)
		}
		if len(item.InternalVector) != 512 {
			return nil, nil, 0, fmt.Errorf("validation failed on item %d (%s): vector has %d dimensions, expected 512", i, trackLabel, len(item.InternalVector))
		}
		if item.Action != "merge" && item.Action != "new_identity" && item.Action != "new_variant" {
			return nil, nil, 0, fmt.Errorf("validation failed on item %d (%s): invalid action '%s'", i, trackLabel, item.Action)
		}
		identity := strings.TrimSpace(stagingItemIdentity(item))
		variant := strings.TrimSpace(stagingItemVariant(item))
		switch item.Action {
		case "merge":
			if identity == "" {
				return nil, nil, 0, fmt.Errorf("validation failed on item %d (%s): action 'merge' requires `identity` to be set", i, trackLabel)
			}
			if variant == "" {
				return nil, nil, 0, fmt.Errorf("validation failed on item %d (%s): action 'merge' requires `variant` to be set", i, trackLabel)
			}
		case "new_variant":
			if identity == "" {
				return nil, nil, 0, fmt.Errorf("validation failed on item %d (%s): action 'new_variant' requires `identity` to be set", i, trackLabel)
			}
			if variant == "" {
				return nil, nil, 0, fmt.Errorf("validation failed on item %d (%s): action 'new_variant' requires `variant` to be set to the new variant name", i, trackLabel)
			}
		case "new_identity":
			if variant != "" {
				return nil, nil, 0, fmt.Errorf("validation failed on item %d (%s): action 'new_identity' requires `variant` to be blank; Sentinel will create the `Default` variant", i, trackLabel)
			}
		}
		if item.InternalCount <= 0 {
			return nil, nil, 0, fmt.Errorf("validation failed on item %d (%s): internal_count must be positive, got %d", i, trackLabel, item.InternalCount)
		}
		if review.VideoID != "" {
			if item.StartTime < 0 || item.EndTime < 0 {
				return nil, nil, 0, fmt.Errorf("validation failed on item %d (%s): interval times must be non-negative", i, trackLabel)
			}
			if item.EndTime < item.StartTime {
				return nil, nil, 0, fmt.Errorf("validation failed on item %d (%s): end_time must be >= start_time", i, trackLabel)
			}
		}
		if trackKey == "" {
			return nil, nil, 0, fmt.Errorf("validation failed on item %d: missing review track id", i)
		}

		actions = append(actions, store.CommitAction{
			TrackID:      trackKey,
			Action:       item.Action,
			IdentityName: stagingItemIdentity(item),
			VariantName:  stagingItemVariant(item),
			Vector:       item.InternalVector,
			Count:        item.InternalCount,
		})
		if review.VideoID != "" {
			intervals = append(intervals, store.CommitInterval{
				TrackID:   trackKey,
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

func hydrateReviewDocument(stagingPath string, review ReviewDocument) (ReviewDocument, error) {
	if !reviewNeedsSidecar(review) {
		return review, nil
	}

	sidecarPath := reviewDataFilePath(stagingPath)
	sidecar, err := readReviewSidecarFile(sidecarPath)
	if err != nil {
		if os.IsNotExist(err) {
			return ReviewDocument{}, fmt.Errorf("review data file not found: %s", sidecarPath)
		}
		return ReviewDocument{}, fmt.Errorf("failed to read review data file: %w", err)
	}

	var missing []string
	for i := range review.Tracks {
		item := &review.Tracks[i]
		if !trackNeedsInternalData(*item) || trackHasInternalData(*item) {
			continue
		}
		key := stagingItemKey(*item)
		data, ok := sidecar.Tracks[key]
		if !ok {
			missing = append(missing, stagingItemLabel(*item))
			continue
		}
		item.InternalVector = data.InternalVector
		item.InternalCount = data.InternalCount
	}

	if len(missing) > 0 {
		sort.Strings(missing)
		return ReviewDocument{}, fmt.Errorf("review data file %s is missing track data for: %s", sidecarPath, strings.Join(missing, ", "))
	}

	return review, nil
}

func reviewNeedsSidecar(review ReviewDocument) bool {
	for _, item := range review.Tracks {
		if trackNeedsInternalData(item) && !trackHasInternalData(item) {
			return true
		}
	}
	return false
}

func trackNeedsInternalData(item StagingItem) bool {
	return item.Action != "" && item.Action != "discard"
}

func trackHasInternalData(item StagingItem) bool {
	return len(item.InternalVector) > 0 && item.InternalCount > 0
}
