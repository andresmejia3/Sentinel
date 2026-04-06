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
	groupedCreateItems := make(map[int][]StagingItem)
	groupedCreateHandled := make(map[int]struct{})
	groupIndexes := make(map[string]int, len(review.Tracks))

	for i, item := range review.Tracks {
		action := strings.TrimSpace(item.Action)
		if item.GroupID > 0 && (action == "new_identity" || action == "new_variant") {
			groupedCreateItems[item.GroupID] = append(groupedCreateItems[item.GroupID], item)
		}
		if key := stagingItemKey(item); key != "" {
			groupIndexes[key] = i
		}
	}

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
		if item.GroupID > 0 && (action == "new_identity" || action == "new_variant") {
			if _, ok := groupedCreateHandled[item.GroupID]; ok {
				continue
			}
			groupedCreateHandled[item.GroupID] = struct{}{}
			identity := strings.TrimSpace(item.Identity)
			variant := strings.TrimSpace(item.Variant)
			var err error
			actions, intervals, err = appendGroupedCreateActions(actions, intervals, groupedCreateItems[item.GroupID], groupIndexes, action, identity, variant)
			if err != nil {
				return nil, nil, 0, err
			}
			continue
		}
		identity := strings.TrimSpace(item.Identity)
		variant := strings.TrimSpace(item.Variant)
		if err := validatePreparedCommitItem(item, i, action, identity, variant, ""); err != nil {
			return nil, nil, 0, err
		}

		actions = append(actions, buildPreparedCommitAction(item, action, identity, variant, ""))
		intervals = append(intervals, buildPreparedCommitInterval(item))
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

func parseGroupedTrackRef(ref string) (string, int, error) {
	trimmed := strings.TrimSpace(ref)
	if trimmed == "" {
		return "", 0, fmt.Errorf("blank track reference")
	}
	id, err := strconv.Atoi(trimmed)
	if err != nil || id <= 0 {
		return "", 0, fmt.Errorf("invalid track reference %q", ref)
	}
	canonical := strconv.Itoa(id)
	if canonical != trimmed {
		return "", 0, fmt.Errorf("track reference %q must use canonical integer form", ref)
	}
	return canonical, id, nil
}

func expandGroupedTrackRefs(ref string, availableTrackIDs []int) ([]string, error) {
	trimmed := strings.TrimSpace(ref)
	if trimmed == "" {
		return nil, fmt.Errorf("blank track reference")
	}
	if !strings.Contains(trimmed, "-") {
		key, _, err := parseGroupedTrackRef(trimmed)
		if err != nil {
			return nil, err
		}
		return []string{key}, nil
	}

	parts := strings.Split(trimmed, "-")
	if len(parts) != 2 {
		return nil, fmt.Errorf("invalid track range %q", ref)
	}

	startKey, startID, err := parseGroupedTrackRef(parts[0])
	if err != nil {
		return nil, fmt.Errorf("invalid track range %q: %w", ref, err)
	}
	endKey, endID, err := parseGroupedTrackRef(parts[1])
	if err != nil {
		return nil, fmt.Errorf("invalid track range %q: %w", ref, err)
	}
	if canonical := startKey + "-" + endKey; canonical != trimmed {
		return nil, fmt.Errorf("track range %q must use canonical integer form like %q", ref, canonical)
	}
	if startID > endID {
		return nil, fmt.Errorf("track range %q is descending; use ascending inclusive ranges", ref)
	}

	startIdx := sort.SearchInts(availableTrackIDs, startID)
	endIdx := sort.Search(len(availableTrackIDs), func(i int) bool {
		return availableTrackIDs[i] > endID
	})

	expected := startID
	capHint := endIdx - startIdx
	if capHint < 0 {
		capHint = 0
	}
	keys := make([]string, 0, capHint)
	for _, id := range availableTrackIDs[startIdx:endIdx] {
		if id != expected {
			return nil, fmt.Errorf("track %q is not defined under raw_tracks", strconv.Itoa(expected))
		}
		keys = append(keys, strconv.Itoa(id))
		expected++
	}
	if expected != endID+1 {
		return nil, fmt.Errorf("track %q is not defined under raw_tracks", strconv.Itoa(expected))
	}
	return keys, nil
}

func expandGroupedReviewDocument(fileDoc ReviewFileDocument) (ReviewDocument, error) {
	review := ReviewDocument{
		ReviewID:  fileDoc.ReviewID,
		VideoID:   fileDoc.VideoID,
		InputPath: fileDoc.InputPath,
		Tracks:    make([]StagingItem, 0, len(fileDoc.RawTracks)),
	}

	rawTracks := make(map[string]ReviewRawTrackItem, len(fileDoc.RawTracks))
	availableTrackIDs := make([]int, 0, len(fileDoc.RawTracks))
	for key, item := range fileDoc.RawTracks {
		canonical, id, err := parseGroupedTrackRef(key)
		if err != nil {
			return ReviewDocument{}, fmt.Errorf("validation failed on raw_tracks key %q: %w", key, err)
		}
		rawTracks[canonical] = item
		availableTrackIDs = append(availableTrackIDs, id)
	}
	sort.Ints(availableTrackIDs)

	unresolvedTrackRefs := make(map[string]struct{}, len(fileDoc.UnresolvedTracks))
	for i, ref := range fileDoc.UnresolvedTracks {
		key, _, err := parseGroupedTrackRef(ref)
		if err != nil {
			return ReviewDocument{}, fmt.Errorf("validation failed on unresolved_tracks[%d]: %w", i, err)
		}
		if _, ok := rawTracks[key]; !ok {
			return ReviewDocument{}, fmt.Errorf("validation failed on unresolved_tracks[%d]: track %q is not defined under raw_tracks", i, key)
		}
		if _, exists := unresolvedTrackRefs[key]; exists {
			return ReviewDocument{}, fmt.Errorf("validation failed on unresolved_tracks[%d]: track %q is listed more than once", i, key)
		}
		unresolvedTrackRefs[key] = struct{}{}
	}

	seenPotentialIdentityIDs := make(map[int]int, len(fileDoc.PotentialIdentities))
	assignedTracks := make(map[string]int, len(rawTracks))

	for i, identity := range fileDoc.PotentialIdentities {
		if identity.ID <= 0 {
			return ReviewDocument{}, fmt.Errorf("validation failed on potential identity %d: missing positive `id`", i+1)
		}
		if prev, ok := seenPotentialIdentityIDs[identity.ID]; ok {
			return ReviewDocument{}, fmt.Errorf("review file has duplicate potential identity id '%d' on items %d and %d", identity.ID, prev, i+1)
		}
		seenPotentialIdentityIDs[identity.ID] = i + 1
		if len(identity.Tracks) == 0 {
			return ReviewDocument{}, fmt.Errorf("validation failed on potential identity %d: `tracks` must contain at least one <track_id> or <start>-<end> range", identity.ID)
		}

		seenTracksInIdentity := make(map[string]struct{}, len(identity.Tracks))
		for _, ref := range identity.Tracks {
			keys, err := expandGroupedTrackRefs(ref, availableTrackIDs)
			if err != nil {
				return ReviewDocument{}, fmt.Errorf("validation failed on potential identity %d: %w", identity.ID, err)
			}
			for _, key := range keys {
				if _, ok := seenTracksInIdentity[key]; ok {
					return ReviewDocument{}, fmt.Errorf("validation failed on potential identity %d: track %q is listed more than once", identity.ID, key)
				}
				seenTracksInIdentity[key] = struct{}{}

				id, err := strconv.Atoi(key)
				if err != nil {
					return ReviewDocument{}, fmt.Errorf("validation failed on potential identity %d: invalid expanded track id %q", identity.ID, key)
				}
				rawTrack, ok := rawTracks[key]
				if !ok {
					return ReviewDocument{}, fmt.Errorf("validation failed on potential identity %d: track %q is not defined under raw_tracks", identity.ID, key)
				}
				if prevIdentityID, ok := assignedTracks[key]; ok {
					return ReviewDocument{}, fmt.Errorf("review file assigns track %q to multiple potential identities (%d and %d)", key, prevIdentityID, identity.ID)
				}
				assignedTracks[key] = identity.ID

				review.Tracks = append(review.Tracks, StagingItem{
					ID:                id,
					StartTime:         rawTrack.StartTime,
					EndTime:           rawTrack.EndTime,
					NearestCandidates: append([]NearestCandidate(nil), rawTrack.NearestCandidates...),
					Confidence:        rawTrack.Confidence,
					Identity:          identity.Identity,
					Variant:           identity.Variant,
					Action:            identity.Action,
					GroupID:           identity.ID,
				})
			}
		}
	}

	var unassigned []string
	for key := range rawTracks {
		if _, ok := assignedTracks[key]; !ok {
			unassigned = append(unassigned, key)
		}
	}
	if len(unassigned) > 0 {
		sort.Strings(unassigned)
		allUnassignedMarkedUnresolved := true
		for _, key := range unassigned {
			if _, ok := unresolvedTrackRefs[key]; !ok {
				allUnassignedMarkedUnresolved = false
				break
			}
		}
		if allUnassignedMarkedUnresolved {
			return ReviewDocument{}, fmt.Errorf("review file has unresolved_tracks that must be moved into potential_identities before commit: %s", strings.Join(unassigned, ", "))
		}
		return ReviewDocument{}, fmt.Errorf("review file has raw_tracks that are not assigned to any potential identity: %s", strings.Join(unassigned, ", "))
	}

	return review, nil
}

func validatePreparedCommitItem(item StagingItem, index int, action, identity, variant, targetTrackID string) error {
	trackLabel := stagingItemLabel(item)
	if trackLabel == "" {
		trackLabel = fmt.Sprintf("item_%d", index)
	}
	if len(item.InternalVector) != 512 {
		return fmt.Errorf("validation failed on item %d (%s): vector has %d dimensions, expected 512", index, trackLabel, len(item.InternalVector))
	}
	switch action {
	case "merge":
		if targetTrackID == "" {
			if identity == "" {
				return fmt.Errorf("validation failed on item %d (%s): action 'merge' requires `identity` to be set", index, trackLabel)
			}
			if variant == "" {
				return fmt.Errorf("validation failed on item %d (%s): action 'merge' requires `variant` to be set", index, trackLabel)
			}
		}
	case "new_variant":
		if identity == "" {
			return fmt.Errorf("validation failed on item %d (%s): action 'new_variant' requires `identity` to be set", index, trackLabel)
		}
		if variant == "" {
			return fmt.Errorf("validation failed on item %d (%s): action 'new_variant' requires `variant` to be set to the new variant name", index, trackLabel)
		}
	case "new_identity":
		if variant != "" {
			return fmt.Errorf("validation failed on item %d (%s): action 'new_identity' requires `variant` to be blank; Sentinel will create the `Default` variant", index, trackLabel)
		}
	default:
		return fmt.Errorf("validation failed on item %d (%s): invalid action '%s'", index, trackLabel, action)
	}
	if item.InternalCount <= 0 {
		return fmt.Errorf("validation failed on item %d (%s): internal_count must be positive, got %d", index, trackLabel, item.InternalCount)
	}
	if item.StartTime < 0 || item.EndTime < 0 {
		return fmt.Errorf("validation failed on item %d (%s): interval times must be non-negative", index, trackLabel)
	}
	if item.EndTime < item.StartTime {
		return fmt.Errorf("validation failed on item %d (%s): end_time must be >= start_time", index, trackLabel)
	}
	if stagingItemKey(item) == "" {
		return fmt.Errorf("validation failed on item %d: missing review track id", index)
	}
	return nil
}

func buildPreparedCommitAction(item StagingItem, action, identity, variant, targetTrackID string) store.CommitAction {
	return store.CommitAction{
		TrackID:       stagingItemKey(item),
		Action:        action,
		IdentityName:  identity,
		VariantName:   variant,
		Vector:        item.InternalVector,
		Count:         item.InternalCount,
		TargetTrackID: targetTrackID,
	}
}

func buildPreparedCommitInterval(item StagingItem) store.CommitInterval {
	return store.CommitInterval{
		TrackID:   stagingItemKey(item),
		Start:     item.StartTime,
		End:       item.EndTime,
		FaceCount: item.InternalCount,
	}
}

func appendGroupedCreateActions(
	actions []store.CommitAction,
	intervals []store.CommitInterval,
	groupItems []StagingItem,
	groupIndexes map[string]int,
	action string,
	identity string,
	variant string,
) ([]store.CommitAction, []store.CommitInterval, error) {
	if len(groupItems) == 0 {
		return actions, intervals, nil
	}

	for _, item := range groupItems {
		idx := groupIndexes[stagingItemKey(item)]
		if strings.TrimSpace(item.Action) != action || strings.TrimSpace(item.Identity) != identity || strings.TrimSpace(item.Variant) != variant {
			return nil, nil, fmt.Errorf("validation failed on potential identity %d: all member tracks must share the same action/identity/variant", item.GroupID)
		}
		if err := validatePreparedCommitItem(item, idx, action, identity, variant, ""); err != nil {
			return nil, nil, err
		}
	}

	seed := groupItems[0]
	actions = append(actions, buildPreparedCommitAction(seed, action, identity, variant, ""))
	intervals = append(intervals, buildPreparedCommitInterval(seed))

	seedTrackID := stagingItemKey(seed)
	for _, item := range groupItems[1:] {
		idx := groupIndexes[stagingItemKey(item)]
		if err := validatePreparedCommitItem(item, idx, "merge", "", "", seedTrackID); err != nil {
			return nil, nil, err
		}
		actions = append(actions, buildPreparedCommitAction(item, "merge", "", "", seedTrackID))
		intervals = append(intervals, buildPreparedCommitInterval(item))
	}

	return actions, intervals, nil
}

func readReviewDocument(data []byte) (ReviewDocument, error) {
	var fileDoc ReviewFileDocument
	dec := yaml.NewDecoder(bytes.NewReader(data))
	dec.KnownFields(true)
	if err := dec.Decode(&fileDoc); err != nil {
		return ReviewDocument{}, fmt.Errorf("failed to parse review YAML: expected current grouped review document format with top-level raw_tracks, optional unresolved_tracks, and potential_identities: %w", err)
	}
	if fileDoc.RawTracks == nil && fileDoc.PotentialIdentities == nil && fileDoc.VideoID == "" && fileDoc.ReviewID == "" && fileDoc.InputPath == "" {
		return ReviewDocument{}, fmt.Errorf("failed to parse review YAML: expected current grouped review document format with top-level raw_tracks, optional unresolved_tracks, and potential_identities")
	}
	return expandGroupedReviewDocument(fileDoc)
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
