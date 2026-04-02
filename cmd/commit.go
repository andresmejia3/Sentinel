package cmd

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/andresmejia3/sentinel/internal/store"
	"github.com/google/uuid"
	"github.com/spf13/cobra"
	"gopkg.in/yaml.v3"
)

var systemIdentityLabelPattern = regexp.MustCompile(`(?i)^identity (\d+)$`)

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

	review, err := readReviewDocument(data)
	if err != nil {
		return err
	}
	if reviewDocumentID(review) == "" {
		return fmt.Errorf("review file is missing review_id")
	}
	if review.VideoID == "" {
		return fmt.Errorf("review file is missing video_id")
	}
	if review.InputPath == "" {
		return fmt.Errorf("review file is missing input_path")
	}
	if err := validateReviewTrackKeys(review); err != nil {
		return err
	}
	review, err = hydrateReviewDocument(stagingPath, review)
	if err != nil {
		return err
	}
	fmt.Println("🔍 Validating review file...")
	actions, intervals, skipCount, err := prepareCommitBatch(review)
	if err != nil {
		return err
	}
	if err := validateCommitActionsWithLookup(ctx, actions, DB.GetIdentityIDByName, DB.IdentityExists); err != nil {
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

func prepareCommitBatch(review ReviewDocument) ([]store.CommitAction, []store.CommitInterval, int, error) {
	if reviewDocumentID(review) == "" {
		return nil, nil, 0, fmt.Errorf("review file is missing review_id")
	}
	if review.VideoID == "" {
		return nil, nil, 0, fmt.Errorf("review file is missing video_id")
	}
	if review.InputPath == "" {
		return nil, nil, 0, fmt.Errorf("review file is missing input_path")
	}
	if err := validateReviewTrackKeys(review); err != nil {
		return nil, nil, 0, err
	}

	var actions []store.CommitAction
	var intervals []store.CommitInterval
	skipCount := 0
	var unresolvedTracks []string

	for i, item := range review.Tracks {
		action := strings.TrimSpace(item.Action)
		if action == "" {
			trackID := stagingItemLabel(item)
			if trackID == "" {
				trackID = fmt.Sprintf("item_%d", i)
			}
			unresolvedTracks = append(unresolvedTracks, trackID)
			continue
		}
		if action == "discard" {
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
		if action != "merge" && action != "new_identity" && action != "new_variant" {
			return nil, nil, 0, fmt.Errorf("validation failed on item %d (%s): invalid action '%s'", i, trackLabel, strings.TrimSpace(item.Action))
		}
		identity := strings.TrimSpace(item.Identity)
		variant := strings.TrimSpace(item.Variant)
		switch action {
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
		if item.StartTime < 0 || item.EndTime < 0 {
			return nil, nil, 0, fmt.Errorf("validation failed on item %d (%s): interval times must be non-negative", i, trackLabel)
		}
		if item.EndTime < item.StartTime {
			return nil, nil, 0, fmt.Errorf("validation failed on item %d (%s): end_time must be >= start_time", i, trackLabel)
		}
		if trackKey == "" {
			return nil, nil, 0, fmt.Errorf("validation failed on item %d: missing review track id", i)
		}

		actions = append(actions, store.CommitAction{
			TrackID:      trackKey,
			Action:       action,
			IdentityName: identity,
			VariantName:  variant,
			Vector:       item.InternalVector,
			Count:        item.InternalCount,
		})
		intervals = append(intervals, store.CommitInterval{
			TrackID:   trackKey,
			Start:     item.StartTime,
			End:       item.EndTime,
			FaceCount: item.InternalCount,
		})
	}

	if len(unresolvedTracks) > 0 {
		return nil, nil, 0, fmt.Errorf("review file has unresolved tracks with blank action: %s", strings.Join(unresolvedTracks, ", "))
	}

	return actions, intervals, skipCount, nil
}

func validateCommitActionsWithLookup(
	ctx context.Context,
	actions []store.CommitAction,
	lookupIdentityID func(context.Context, string) (int, error),
	identityExists func(context.Context, int) (bool, error),
) error {
	seenNewIdentityNames := make(map[string]string, len(actions))

	for _, action := range actions {
		if action.Action != "new_identity" || action.IdentityName == "" {
			continue
		}
		canonicalName := canonicalizeReviewName(action.IdentityName)
		if existingTrackID, ok := seenNewIdentityNames[canonicalName]; ok {
			return fmt.Errorf("validation failed on track %s: identity name %q is duplicated in this review batch (also used by track %s). Use 'merge'/'new_variant' if these tracks are the same person, or choose distinct new identity names", action.TrackID, action.IdentityName, existingTrackID)
		}
		seenNewIdentityNames[canonicalName] = action.TrackID

		if matches := systemIdentityLabelPattern.FindStringSubmatch(action.IdentityName); len(matches) == 2 {
			parsedID, _ := strconv.Atoi(matches[1])
			exists, err := identityExists(ctx, parsedID)
			if err != nil {
				return fmt.Errorf("failed to validate identity name %q for track %s: %w", action.IdentityName, action.TrackID, err)
			}
			if exists {
				return fmt.Errorf("validation failed on track %s: identity name %q conflicts with Sentinel's system label for existing identity %d. Choose a different new identity name, or use 'merge'/'new_variant' if this is the same person", action.TrackID, action.IdentityName, parsedID)
			}
		}

		existingID, err := lookupIdentityID(ctx, action.IdentityName)
		if err != nil {
			return fmt.Errorf("failed to validate identity name %q for track %s: %w", action.IdentityName, action.TrackID, err)
		}
		if existingID != 0 {
			return fmt.Errorf("validation failed on track %s: identity name %q already exists (identity %d). If this is the same person, use 'merge' or 'new_variant' instead; otherwise choose a different new identity name", action.TrackID, action.IdentityName, existingID)
		}
	}

	return nil
}

func canonicalizeReviewName(name string) string {
	return strings.ToLower(strings.TrimSpace(name))
}

func validateReviewTrackKeys(review ReviewDocument) error {
	seen := make(map[string]int, len(review.Tracks))
	for i, item := range review.Tracks {
		key := stagingItemKey(item)
		if key == "" {
			return fmt.Errorf("validation failed on item %d: missing review track id", i)
		}
		if prev, ok := seen[key]; ok {
			return fmt.Errorf("review file has duplicate track id '%s' on items %d and %d", key, prev, i+1)
		}
		seen[key] = i + 1
	}
	return nil
}

func readReviewDocument(data []byte) (ReviewDocument, error) {
	var review ReviewDocument
	dec := yaml.NewDecoder(bytes.NewReader(data))
	dec.KnownFields(true)
	if err := dec.Decode(&review); err != nil {
		return ReviewDocument{}, fmt.Errorf("failed to parse review YAML: expected current review document format with top-level tracks: %w", err)
	}
	if review.Tracks == nil && review.VideoID == "" && reviewDocumentID(review) == "" && review.InputPath == "" {
		return ReviewDocument{}, fmt.Errorf("failed to parse review YAML: expected current review document format with top-level tracks")
	}
	return review, nil
}

func hydrateReviewDocument(stagingPath string, review ReviewDocument) (ReviewDocument, error) {
	if err := validateReviewHasNoEmbeddedInternalData(review); err != nil {
		return ReviewDocument{}, err
	}
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
	if err := validateReviewSidecarMetadata(review, sidecarPath, sidecar); err != nil {
		return ReviewDocument{}, err
	}
	if err := validateReviewSidecarTrackKeys(review, sidecarPath, sidecar); err != nil {
		return ReviewDocument{}, err
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
		expectedFingerprint, err := reviewTrackFingerprint(*item)
		if err != nil {
			return ReviewDocument{}, err
		}
		if data.Fingerprint == "" {
			return ReviewDocument{}, fmt.Errorf("review data file %s is missing fingerprint for track %s", sidecarPath, stagingItemLabel(*item))
		}
		if data.Fingerprint != expectedFingerprint {
			return ReviewDocument{}, fmt.Errorf("review data file %s fingerprint mismatch for track %s; the review's read-only track fields may have been edited", sidecarPath, stagingItemLabel(*item))
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

func validateReviewSidecarTrackKeys(review ReviewDocument, sidecarPath string, sidecar ReviewSidecarDocument) error {
	expected := make(map[string]struct{}, len(review.Tracks))
	for i, item := range review.Tracks {
		key := stagingItemKey(item)
		if key == "" {
			return fmt.Errorf("validation failed on item %d: missing review track id", i)
		}
		expected[key] = struct{}{}
	}

	var unexpected []string
	for key := range sidecar.Tracks {
		if _, ok := expected[key]; !ok {
			unexpected = append(unexpected, key)
		}
	}
	var missing []string
	for key := range expected {
		if _, ok := sidecar.Tracks[key]; !ok {
			missing = append(missing, key)
		}
	}
	if len(unexpected) > 0 || len(missing) > 0 {
		sort.Strings(unexpected)
		sort.Strings(missing)
		switch {
		case len(unexpected) > 0 && len(missing) > 0:
			return fmt.Errorf("review data file %s track set mismatch: unexpected track data for %s; missing track data for %s", sidecarPath, strings.Join(unexpected, ", "), strings.Join(missing, ", "))
		case len(unexpected) > 0:
			return fmt.Errorf("review data file %s has unexpected track data for: %s", sidecarPath, strings.Join(unexpected, ", "))
		default:
			return fmt.Errorf("review data file %s is missing track data for: %s", sidecarPath, strings.Join(missing, ", "))
		}
	}

	return nil
}

func validateReviewSidecarMetadata(review ReviewDocument, sidecarPath string, sidecar ReviewSidecarDocument) error {
	reviewID := reviewDocumentID(review)
	sidecarID := sidecarDocumentID(sidecar)
	if reviewID != "" {
		if sidecarID == "" {
			return fmt.Errorf("review data file %s is missing review_id", sidecarPath)
		}
		if sidecarID != reviewID {
			return fmt.Errorf("review data file %s has review_id %q, expected %q", sidecarPath, sidecarID, reviewID)
		}
	}
	if review.VideoID != "" {
		if sidecar.VideoID == "" {
			return fmt.Errorf("review data file %s is missing video_id", sidecarPath)
		}
		if sidecar.VideoID != review.VideoID {
			return fmt.Errorf("review data file %s has video_id %q, expected %q", sidecarPath, sidecar.VideoID, review.VideoID)
		}
	}
	if review.InputPath != "" {
		if sidecar.InputPath == "" {
			return fmt.Errorf("review data file %s is missing input_path", sidecarPath)
		}
		if sidecar.InputPath != review.InputPath {
			return fmt.Errorf("review data file %s has input_path %q, expected %q", sidecarPath, sidecar.InputPath, review.InputPath)
		}
	}
	return nil
}

func reviewNeedsSidecar(review ReviewDocument) bool {
	if len(review.Tracks) == 0 {
		return true
	}
	for _, item := range review.Tracks {
		if !trackHasInternalData(item) {
			return true
		}
	}
	return false
}

func validateReviewHasNoEmbeddedInternalData(review ReviewDocument) error {
	for i, item := range review.Tracks {
		if trackHasInternalData(item) {
			trackLabel := stagingItemLabel(item)
			if trackLabel == "" {
				trackLabel = fmt.Sprintf("item_%d", i)
			}
			return fmt.Errorf("review file must not embed internal data for track %s; internal_vector/internal_count must come from the sibling review data file", trackLabel)
		}
	}
	return nil
}

func trackNeedsInternalData(item StagingItem) bool {
	return item.Action != "" && item.Action != "discard"
}

func trackHasInternalData(item StagingItem) bool {
	return len(item.InternalVector) > 0 && item.InternalCount > 0
}
