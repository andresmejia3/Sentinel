package cmd

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/andresmejia3/sentinel/internal/store"
)

func TestReadReviewDocumentExpandsGroupedCurrentFormat(t *testing.T) {
	data := []byte(`
review_id: a1b2c3d4e5f6
video_id: vid_test
input_path: samples/test.mp4
raw_tracks:
  "1":
    start_time: 0
    end_time: 2
    nearest_candidates:
      - identity: Jenny
        variant: Default
        distance: 0.21
    confidence: 0.89
    suggested_identity: Jenny
    suggested_variant: Default
    suggested_action: merge
    artifact_dir: results/vid_test/reviews/a1b2c3d4e5f6/tracks/1
  "2":
    start_time: 3
    end_time: 5
    confidence: 0.78
    suggested_identity: Jenny
    suggested_variant: Default
    suggested_action: merge
potential_identities:
  - id: 1
    tracks:
      - "1"
      - "2"
    identity: Jenny
    variant: Default
    action: merge
`)

	review, err := readReviewDocument(data)
	if err != nil {
		t.Fatalf("readReviewDocument returned error: %v", err)
	}
	if review.ReviewID != "a1b2c3d4e5f6" || review.VideoID != "vid_test" || review.InputPath != "samples/test.mp4" {
		t.Fatalf("unexpected review metadata: %+v", review)
	}
	if len(review.Tracks) != 2 || review.Tracks[0].ID != 1 || review.Tracks[1].ID != 2 {
		t.Fatalf("unexpected review tracks: %+v", review.Tracks)
	}
	if got := review.Tracks[0].Identity; got != "Jenny" {
		t.Fatalf("expected grouped identity to expand onto track, got %q", got)
	}
	if got := review.Tracks[0].Variant; got != "Default" {
		t.Fatalf("expected grouped variant to expand onto track, got %q", got)
	}
	if got := review.Tracks[0].Action; got != "merge" {
		t.Fatalf("expected grouped action to expand onto track, got %q", got)
	}
	if got := review.Tracks[0].NearestCandidates[0].Identity; got != "Jenny" {
		t.Fatalf("expected raw-track nearest candidates to be preserved, got %+v", review.Tracks[0].NearestCandidates)
	}
}

func TestReadReviewDocumentRejectsLegacyListFormat(t *testing.T) {
	data := []byte(`
- id: 1
  action: merge
`)

	_, err := readReviewDocument(data)
	if err == nil {
		t.Fatal("expected legacy bare-list review format to be rejected")
	}
	if !strings.Contains(err.Error(), "current grouped review document format") {
		t.Fatalf("expected current-format guidance in error, got: %v", err)
	}
}

func TestReadReviewDocumentRejectsEmbeddedInternalDataFields(t *testing.T) {
	data := []byte(`
review_id: a1b2c3d4e5f6
video_id: vid_test
input_path: samples/test.mp4
raw_tracks:
  "1":
    start_time: 0
    end_time: 2
    internal_vector: [1, 2, 3]
potential_identities:
  - id: 1
    tracks:
      - "1"
    action: merge
`)

	_, err := readReviewDocument(data)
	if err == nil {
		t.Fatal("expected embedded internal data fields to be rejected")
	}
	if !strings.Contains(err.Error(), "internal_vector") {
		t.Fatalf("expected unknown internal_vector field in error, got: %v", err)
	}
}

func TestReadReviewDocumentRejectsDuplicateTrackAssignmentAcrossPotentialIdentities(t *testing.T) {
	data := []byte(`
review_id: a1b2c3d4e5f6
video_id: vid_test
input_path: samples/test.mp4
raw_tracks:
  "1":
    start_time: 0
    end_time: 2
    confidence: 0.81
potential_identities:
  - id: 1
    tracks:
      - "1"
    action: discard
  - id: 2
    tracks:
      - "1"
    action: discard
`)

	_, err := readReviewDocument(data)
	if err == nil {
		t.Fatal("expected duplicate grouped track assignment to be rejected")
	}
	if !strings.Contains(err.Error(), "multiple potential identities") {
		t.Fatalf("expected duplicate grouped track error, got: %v", err)
	}
}

func TestReadReviewDocumentRejectsDuplicatePotentialIdentityIDs(t *testing.T) {
	data := []byte(`
review_id: a1b2c3d4e5f6
video_id: vid_test
input_path: samples/test.mp4
raw_tracks:
  "1":
    start_time: 0
    end_time: 2
    confidence: 0.81
  "2":
    start_time: 3
    end_time: 5
    confidence: 0.77
potential_identities:
  - id: 1
    tracks:
      - "1"
    action: discard
  - id: 1
    tracks:
      - "2"
    action: discard
`)

	_, err := readReviewDocument(data)
	if err == nil {
		t.Fatal("expected duplicate potential identity IDs to be rejected")
	}
	if !strings.Contains(err.Error(), "duplicate potential identity id '1'") {
		t.Fatalf("expected duplicate potential identity ID error, got: %v", err)
	}
}

func TestReadReviewDocumentRejectsUnassignedRawTracks(t *testing.T) {
	data := []byte(`
review_id: a1b2c3d4e5f6
video_id: vid_test
input_path: samples/test.mp4
raw_tracks:
  "1":
    start_time: 0
    end_time: 2
    confidence: 0.81
  "2":
    start_time: 3
    end_time: 5
    confidence: 0.77
potential_identities:
  - id: 1
    tracks:
      - "1"
    action: discard
`)

	_, err := readReviewDocument(data)
	if err == nil {
		t.Fatal("expected unassigned raw_tracks to be rejected")
	}
	if !strings.Contains(err.Error(), "not assigned to any potential identity") {
		t.Fatalf("expected unassigned raw_tracks error, got: %v", err)
	}
}

func TestReadReviewDocumentRejectsUnresolvedTracksThatWereNotMovedIntoPotentialIdentities(t *testing.T) {
	data := []byte(`
review_id: a1b2c3d4e5f6
video_id: vid_test
input_path: samples/test.mp4
raw_tracks:
  "1":
    start_time: 0
    end_time: 2
    confidence: 0.81
  "2":
    start_time: 3
    end_time: 5
    confidence: 0.77
unresolved_tracks:
  - "2"
potential_identities:
  - id: 1
    tracks:
      - "1"
    action: discard
`)

	_, err := readReviewDocument(data)
	if err == nil {
		t.Fatal("expected unresolved tracks that were not moved into potential identities to be rejected")
	}
	if !strings.Contains(err.Error(), "unresolved_tracks that must be moved into potential_identities") {
		t.Fatalf("expected unresolved_tracks guidance in error, got: %v", err)
	}
}

func TestReadReviewDocumentAllowsResolvedTrackToRemainListedUnderUnresolvedTracks(t *testing.T) {
	data := []byte(`
review_id: a1b2c3d4e5f6
video_id: vid_test
input_path: samples/test.mp4
raw_tracks:
  "1":
    start_time: 0
    end_time: 2
    confidence: 0.81
  "2":
    start_time: 3
    end_time: 5
    confidence: 0.77
unresolved_tracks:
  - "2"
potential_identities:
  - id: 1
    tracks:
      - "1"
      - "2"
    action: discard
`)

	review, err := readReviewDocument(data)
	if err != nil {
		t.Fatalf("expected unresolved_tracks to be informational-only once track 2 is assigned, got error: %v", err)
	}
	if len(review.Tracks) != 2 || review.Tracks[0].ID != 1 || review.Tracks[1].ID != 2 {
		t.Fatalf("unexpected expanded tracks: %+v", review.Tracks)
	}
}

func TestReadReviewDocumentExpandsGroupedTrackRanges(t *testing.T) {
	data := []byte(`
review_id: a1b2c3d4e5f6
video_id: vid_test
input_path: samples/test.mp4
raw_tracks:
  "1":
    start_time: 0
    end_time: 2
    confidence: 0.81
  "2":
    start_time: 3
    end_time: 5
    confidence: 0.77
  "3":
    start_time: 6
    end_time: 8
    confidence: 0.73
potential_identities:
  - id: 1
    tracks:
      - "1-2"
    action: discard
  - id: 2
    tracks:
      - "3"
    action: discard
`)

	review, err := readReviewDocument(data)
	if err != nil {
		t.Fatalf("expected grouped track ranges to expand successfully, got error: %v", err)
	}
	if len(review.Tracks) != 3 {
		t.Fatalf("expected 3 expanded tracks, got %d", len(review.Tracks))
	}
	if review.Tracks[0].ID != 1 || review.Tracks[1].ID != 2 || review.Tracks[2].ID != 3 {
		t.Fatalf("expected range expansion to preserve ascending track order, got %+v", review.Tracks)
	}
	if review.Tracks[0].GroupID != 1 || review.Tracks[1].GroupID != 1 || review.Tracks[2].GroupID != 2 {
		t.Fatalf("expected expanded tracks to preserve potential identity ids, got %+v", review.Tracks)
	}
}

func TestReadReviewDocumentRejectsMalformedTrackRanges(t *testing.T) {
	data := []byte(`
review_id: a1b2c3d4e5f6
video_id: vid_test
input_path: samples/test.mp4
raw_tracks:
  "1":
    start_time: 0
    end_time: 2
    confidence: 0.81
  "2":
    start_time: 3
    end_time: 5
    confidence: 0.77
potential_identities:
  - id: 1
    tracks:
      - "1-2-3"
    action: discard
`)

	_, err := readReviewDocument(data)
	if err == nil {
		t.Fatal("expected malformed grouped track range to be rejected")
	}
	if !strings.Contains(err.Error(), "invalid track range") {
		t.Fatalf("expected malformed track range error, got: %v", err)
	}
}

func TestReadReviewDocumentRejectsDescendingTrackRanges(t *testing.T) {
	data := []byte(`
review_id: a1b2c3d4e5f6
video_id: vid_test
input_path: samples/test.mp4
raw_tracks:
  "1":
    start_time: 0
    end_time: 2
    confidence: 0.81
  "2":
    start_time: 3
    end_time: 5
    confidence: 0.77
potential_identities:
  - id: 1
    tracks:
      - "2-1"
    action: discard
`)

	_, err := readReviewDocument(data)
	if err == nil {
		t.Fatal("expected descending grouped track range to be rejected")
	}
	if !strings.Contains(err.Error(), "descending") {
		t.Fatalf("expected descending track range error, got: %v", err)
	}
}

func TestReadReviewDocumentRejectsDuplicateExpandedTracksWithinPotentialIdentity(t *testing.T) {
	data := []byte(`
review_id: a1b2c3d4e5f6
video_id: vid_test
input_path: samples/test.mp4
raw_tracks:
  "1":
    start_time: 0
    end_time: 2
    confidence: 0.81
  "2":
    start_time: 3
    end_time: 5
    confidence: 0.77
potential_identities:
  - id: 1
    tracks:
      - "1-2"
      - "2"
    action: discard
`)

	_, err := readReviewDocument(data)
	if err == nil {
		t.Fatal("expected duplicate expanded grouped track to be rejected")
	}
	if !strings.Contains(err.Error(), "listed more than once") {
		t.Fatalf("expected duplicate expanded track error, got: %v", err)
	}
}

func TestReadReviewDocumentRejectsEmptyPotentialIdentityTracksWithUpdatedGuidance(t *testing.T) {
	data := []byte(`
review_id: a1b2c3d4e5f6
video_id: vid_test
input_path: samples/test.mp4
raw_tracks:
  "1":
    start_time: 0
    end_time: 2
    confidence: 0.81
potential_identities:
  - id: 1
    tracks: []
    action: discard
`)

	_, err := readReviewDocument(data)
	if err == nil {
		t.Fatal("expected empty grouped track selector list to be rejected")
	}
	if !strings.Contains(err.Error(), "<track_id> or <start>-<end> range") {
		t.Fatalf("expected updated empty tracks guidance, got: %v", err)
	}
}

func TestReadReviewDocumentRejectsUndefinedTrackExpandedFromRange(t *testing.T) {
	data := []byte(`
review_id: a1b2c3d4e5f6
video_id: vid_test
input_path: samples/test.mp4
raw_tracks:
  "1":
    start_time: 0
    end_time: 2
    confidence: 0.81
  "3":
    start_time: 6
    end_time: 8
    confidence: 0.73
potential_identities:
  - id: 1
    tracks:
      - "1-3"
    action: discard
`)

	_, err := readReviewDocument(data)
	if err == nil {
		t.Fatal("expected undefined track expanded from range to be rejected")
	}
	if !strings.Contains(err.Error(), `track "2" is not defined under raw_tracks`) {
		t.Fatalf("expected undefined expanded track error, got: %v", err)
	}
}

func TestReadReviewDocumentRejectsSparseHugeTrackRangeSafely(t *testing.T) {
	data := []byte(`
review_id: a1b2c3d4e5f6
video_id: vid_test
input_path: samples/test.mp4
raw_tracks:
  "1":
    start_time: 0
    end_time: 2
    confidence: 0.81
  "2":
    start_time: 3
    end_time: 5
    confidence: 0.77
potential_identities:
  - id: 1
    tracks:
      - "1-1000000000"
    action: discard
`)

	_, err := readReviewDocument(data)
	if err == nil {
		t.Fatal("expected sparse huge grouped track range to be rejected")
	}
	if !strings.Contains(err.Error(), `track "3" is not defined under raw_tracks`) {
		t.Fatalf("expected sparse huge range to fail on the first missing track, got: %v", err)
	}
}

func TestGroupedReviewCommitFlowExpandsHydratesAndBuildsBatch(t *testing.T) {
	t.Parallel()

	reviewYAML := []byte(`
review_id: a1b2c3d4e5f6
video_id: vid_test
input_path: samples/test.mp4
raw_tracks:
  "1":
    start_time: 0
    end_time: 2
    nearest_candidates:
      - identity: Jenny
        variant: Default
        distance: 0.21
    confidence: 0.89
    suggested_identity: Jenny
    suggested_variant: Default
    suggested_action: merge
  "4":
    start_time: 3
    end_time: 5
    nearest_candidates:
      - identity: Jenny
        variant: Default
        distance: 0.24
    confidence: 0.85
    suggested_identity: Jenny
    suggested_variant: Default
    suggested_action: merge
potential_identities:
  - id: 1
    tracks:
      - "1"
      - "4"
    identity: Jenny
    variant: Default
    action: merge
`)

	tmpDir := t.TempDir()
	reviewPath := filepath.Join(tmpDir, "scan.review.yaml")
	if err := os.WriteFile(reviewPath, reviewYAML, 0644); err != nil {
		t.Fatalf("failed to write grouped review YAML: %v", err)
	}

	review, err := readReviewDocument(reviewYAML)
	if err != nil {
		t.Fatalf("readReviewDocument returned error: %v", err)
	}

	vec1 := make([]float64, 512)
	vec1[0] = 1.0
	vec4 := make([]float64, 512)
	vec4[3] = 1.0

	fingerprint1, err := reviewTrackFingerprint(review.Tracks[0])
	if err != nil {
		t.Fatalf("reviewTrackFingerprint track 1 returned error: %v", err)
	}
	fingerprint4, err := reviewTrackFingerprint(review.Tracks[1])
	if err != nil {
		t.Fatalf("reviewTrackFingerprint track 4 returned error: %v", err)
	}

	if err := writeReviewSidecarFile(reviewDataFilePath(reviewPath), ReviewSidecarDocument{
		ReviewID:  "a1b2c3d4e5f6",
		VideoID:   "vid_test",
		InputPath: "samples/test.mp4",
		Tracks: map[string]ReviewTrackData{
			"1": {
				Fingerprint:    fingerprint1,
				InternalVector: vec1,
				InternalCount:  2,
			},
			"4": {
				Fingerprint:    fingerprint4,
				InternalVector: vec4,
				InternalCount:  3,
			},
		},
	}); err != nil {
		t.Fatalf("writeReviewSidecarFile returned error: %v", err)
	}

	hydrated, err := hydrateReviewDocument(reviewPath, review)
	if err != nil {
		t.Fatalf("hydrateReviewDocument returned error: %v", err)
	}

	actions, intervals, skipCount, err := prepareCommitBatch(hydrated)
	if err != nil {
		t.Fatalf("prepareCommitBatch returned error: %v", err)
	}
	if skipCount != 0 {
		t.Fatalf("expected skipCount 0, got %d", skipCount)
	}
	if len(actions) != 2 || len(intervals) != 2 {
		t.Fatalf("expected 2 actions and 2 intervals, got %d and %d", len(actions), len(intervals))
	}
	if actions[0].TrackID != "1" || actions[1].TrackID != "4" {
		t.Fatalf("expected raw track ids to survive grouped expansion, got %+v", actions)
	}
	for _, action := range actions {
		if action.Action != "merge" || action.IdentityName != "Jenny" || action.VariantName != "Default" {
			t.Fatalf("expected grouped merge decision to apply to every expanded track, got %+v", action)
		}
	}
	if intervals[0].TrackID != "1" || intervals[1].TrackID != "4" {
		t.Fatalf("expected intervals to use raw track ids after grouped expansion, got %+v", intervals)
	}
}

func TestPrepareCommitBatchGroupedNewIdentityCreatesOnceAndMergesFollowers(t *testing.T) {
	vec1 := make([]float64, 512)
	vec1[0] = 1.0
	vec4 := make([]float64, 512)
	vec4[4] = 1.0

	review := ReviewDocument{
		ReviewID:  "a1b2c3d4e5f6",
		VideoID:   "vid_test",
		InputPath: "/tmp/test.mp4",
		Tracks: []StagingItem{
			{
				ID:             1,
				GroupID:        7,
				Action:         "new_identity",
				Identity:       "Monica",
				InternalVector: vec1,
				InternalCount:  2,
				StartTime:      0,
				EndTime:        2,
			},
			{
				ID:             4,
				GroupID:        7,
				Action:         "new_identity",
				Identity:       "Monica",
				InternalVector: vec4,
				InternalCount:  3,
				StartTime:      3,
				EndTime:        5,
			},
		},
	}

	actions, intervals, skipCount, err := prepareCommitBatch(review)
	if err != nil {
		t.Fatalf("prepareCommitBatch returned error: %v", err)
	}
	if skipCount != 0 {
		t.Fatalf("expected skipCount 0, got %d", skipCount)
	}
	if len(actions) != 2 || len(intervals) != 2 {
		t.Fatalf("expected 2 actions and 2 intervals, got %d and %d", len(actions), len(intervals))
	}
	if actions[0].Action != "new_identity" || actions[0].TrackID != "1" {
		t.Fatalf("expected first action to seed the grouped new_identity, got %+v", actions[0])
	}
	if actions[1].Action != "merge" || actions[1].TrackID != "4" || actions[1].TargetTrackID != "1" {
		t.Fatalf("expected second action to merge into the created seed track, got %+v", actions[1])
	}
	if actions[1].IdentityName != "" || actions[1].VariantName != "" {
		t.Fatalf("expected grouped follower merge to target the created track alias, got %+v", actions[1])
	}
}

func TestPrepareCommitBatchGroupedNewVariantCreatesOnceAndMergesFollowers(t *testing.T) {
	vec1 := make([]float64, 512)
	vec1[1] = 1.0
	vec4 := make([]float64, 512)
	vec4[5] = 1.0

	review := ReviewDocument{
		ReviewID:  "a1b2c3d4e5f6",
		VideoID:   "vid_test",
		InputPath: "/tmp/test.mp4",
		Tracks: []StagingItem{
			{
				ID:             1,
				GroupID:        9,
				Action:         "new_variant",
				Identity:       "Jenny",
				Variant:        "Glasses",
				InternalVector: vec1,
				InternalCount:  2,
				StartTime:      0,
				EndTime:        2,
			},
			{
				ID:             4,
				GroupID:        9,
				Action:         "new_variant",
				Identity:       "Jenny",
				Variant:        "Glasses",
				InternalVector: vec4,
				InternalCount:  3,
				StartTime:      3,
				EndTime:        5,
			},
		},
	}

	actions, intervals, skipCount, err := prepareCommitBatch(review)
	if err != nil {
		t.Fatalf("prepareCommitBatch returned error: %v", err)
	}
	if skipCount != 0 {
		t.Fatalf("expected skipCount 0, got %d", skipCount)
	}
	if len(actions) != 2 || len(intervals) != 2 {
		t.Fatalf("expected 2 actions and 2 intervals, got %d and %d", len(actions), len(intervals))
	}
	if actions[0].Action != "new_variant" || actions[0].TrackID != "1" {
		t.Fatalf("expected first action to seed the grouped new_variant, got %+v", actions[0])
	}
	if actions[1].Action != "merge" || actions[1].TrackID != "4" || actions[1].TargetTrackID != "1" {
		t.Fatalf("expected second action to merge into the created seed track, got %+v", actions[1])
	}
}

func TestPrepareCommitBatchRejectsBlankActions(t *testing.T) {
	review := ReviewDocument{
		ReviewID:  "a1b2c3d4e5f6",
		VideoID:   "vid_test",
		InputPath: "/tmp/test.mp4",
		Tracks: []StagingItem{
			{ID: 1, Action: ""},
			{ID: 2, Action: "discard"},
		},
	}

	_, _, _, err := prepareCommitBatch(review)
	if err == nil {
		t.Fatal("expected blank action to be rejected")
	}
	if !strings.Contains(err.Error(), "1") {
		t.Fatalf("expected unresolved track ID in error, got: %v", err)
	}
}

func TestPrepareCommitBatchRejectsDuplicateReviewIDs(t *testing.T) {
	review := ReviewDocument{
		ReviewID:  "a1b2c3d4e5f6",
		VideoID:   "vid_test",
		InputPath: "/tmp/test.mp4",
		Tracks: []StagingItem{
			{ID: 1, Action: "discard"},
			{ID: 1, Action: "discard"},
		},
	}

	_, _, _, err := prepareCommitBatch(review)
	if err == nil {
		t.Fatal("expected duplicate review IDs to be rejected")
	}
	if !strings.Contains(err.Error(), "duplicate track id '1'") {
		t.Fatalf("expected duplicate review ID error, got: %v", err)
	}
}

func TestPrepareCommitBatchRejectsCurrentFormatWithoutVideoMetadata(t *testing.T) {
	review := ReviewDocument{
		ReviewID: "a1b2c3d4e5f6",
		Tracks: []StagingItem{
			{ID: 1, Action: "discard"},
		},
	}

	_, _, _, err := prepareCommitBatch(review)
	if err == nil {
		t.Fatal("expected missing video metadata to be rejected")
	}
	if !strings.Contains(err.Error(), "video_id") {
		t.Fatalf("expected missing video_id error, got: %v", err)
	}
}

func TestPrepareCommitBatchRejectsCurrentFormatWithoutReviewID(t *testing.T) {
	review := ReviewDocument{
		VideoID:   "vid_test",
		InputPath: "/tmp/test.mp4",
		Tracks: []StagingItem{
			{ID: 1, Action: "discard"},
		},
	}

	_, _, _, err := prepareCommitBatch(review)
	if err == nil {
		t.Fatal("expected missing review_id to be rejected")
	}
	if !strings.Contains(err.Error(), "review_id") {
		t.Fatalf("expected missing review_id error, got: %v", err)
	}
}

func TestPrepareCommitBatchUsesGroundTruthIdentityFields(t *testing.T) {
	vec := make([]float64, 512)
	vec[0] = 1.0

	review := ReviewDocument{
		ReviewID:  "a1b2c3d4e5f6",
		VideoID:   "vid_test",
		InputPath: "/tmp/test.mp4",
		Tracks: []StagingItem{
			{
				ID:             1,
				Action:         "new_variant",
				Identity:       "Jenny",
				Variant:        "Side_Profile",
				InternalVector: vec,
				InternalCount:  2,
			},
		},
	}

	actions, _, _, err := prepareCommitBatch(review)
	if err != nil {
		t.Fatalf("expected review to validate, got error: %v", err)
	}
	if got := actions[0].IdentityName; got != "Jenny" {
		t.Fatalf("expected commit identity to come from identity field, got %q", got)
	}
	if got := actions[0].VariantName; got != "Side_Profile" {
		t.Fatalf("expected commit variant to come from variant field, got %q", got)
	}
}

func TestPrepareCommitBatchTrimsGroundTruthIdentityFields(t *testing.T) {
	vec := make([]float64, 512)
	vec[0] = 1.0

	review := ReviewDocument{
		ReviewID:  "a1b2c3d4e5f6",
		VideoID:   "vid_test",
		InputPath: "/tmp/test.mp4",
		Tracks: []StagingItem{
			{
				ID:             1,
				Action:         "merge",
				Identity:       "  Jenny  ",
				Variant:        "  Default  ",
				InternalVector: vec,
				InternalCount:  2,
			},
		},
	}

	actions, _, _, err := prepareCommitBatch(review)
	if err != nil {
		t.Fatalf("expected review to validate, got error: %v", err)
	}
	if got := actions[0].IdentityName; got != "Jenny" {
		t.Fatalf("expected trimmed commit identity, got %q", got)
	}
	if got := actions[0].VariantName; got != "Default" {
		t.Fatalf("expected trimmed commit variant, got %q", got)
	}
}

func TestPrepareCommitBatchTrimsActionField(t *testing.T) {
	vec := make([]float64, 512)
	vec[0] = 1.0

	review := ReviewDocument{
		ReviewID:  "a1b2c3d4e5f6",
		VideoID:   "vid_test",
		InputPath: "/tmp/test.mp4",
		Tracks: []StagingItem{
			{
				ID:             1,
				Action:         "  merge  ",
				Identity:       "Jenny",
				Variant:        "Default",
				InternalVector: vec,
				InternalCount:  2,
			},
		},
	}

	actions, _, _, err := prepareCommitBatch(review)
	if err != nil {
		t.Fatalf("expected trimmed action to validate, got error: %v", err)
	}
	if got := actions[0].Action; got != "merge" {
		t.Fatalf("expected trimmed commit action, got %q", got)
	}
}

func TestPrepareCommitBatchTreatsWhitespaceActionAsBlank(t *testing.T) {
	review := ReviewDocument{
		ReviewID:  "a1b2c3d4e5f6",
		VideoID:   "vid_test",
		InputPath: "/tmp/test.mp4",
		Tracks: []StagingItem{
			{ID: 1, Action: "   "},
		},
	}

	_, _, _, err := prepareCommitBatch(review)
	if err == nil {
		t.Fatal("expected whitespace-only action to be rejected as unresolved")
	}
	if !strings.Contains(err.Error(), "blank action") {
		t.Fatalf("expected unresolved blank action error, got: %v", err)
	}
}

func TestPrepareCommitBatchRejectsNewVariantWithoutVariantName(t *testing.T) {
	vec := make([]float64, 512)
	vec[0] = 1.0

	review := ReviewDocument{
		ReviewID:  "a1b2c3d4e5f6",
		VideoID:   "vid_test",
		InputPath: "/tmp/test.mp4",
		Tracks: []StagingItem{
			{
				ID:             1,
				Action:         "new_variant",
				Identity:       "Jenny",
				Variant:        "",
				InternalVector: vec,
				InternalCount:  2,
			},
		},
	}

	_, _, _, err := prepareCommitBatch(review)
	if err == nil {
		t.Fatal("expected new_variant without variant to be rejected")
	}
	if !strings.Contains(err.Error(), "requires `variant` to be set") {
		t.Fatalf("expected missing variant validation error, got: %v", err)
	}
}

func TestPrepareCommitBatchRejectsNewIdentityWithExplicitVariant(t *testing.T) {
	vec := make([]float64, 512)
	vec[0] = 1.0

	review := ReviewDocument{
		ReviewID:  "a1b2c3d4e5f6",
		VideoID:   "vid_test",
		InputPath: "/tmp/test.mp4",
		Tracks: []StagingItem{
			{
				ID:             1,
				Action:         "new_identity",
				Variant:        "ShouldNotBeSet",
				InternalVector: vec,
				InternalCount:  2,
			},
		},
	}

	_, _, _, err := prepareCommitBatch(review)
	if err == nil {
		t.Fatal("expected new_identity with explicit variant to be rejected")
	}
	if !strings.Contains(err.Error(), "requires `variant` to be blank") {
		t.Fatalf("expected variant-blank validation error, got: %v", err)
	}
}

func TestValidateCommitActionsWithLookupRejectsExistingNewIdentityName(t *testing.T) {
	actions := []store.CommitAction{
		{
			TrackID:      "1",
			Action:       "new_identity",
			IdentityName: "Jenny",
		},
	}

	err := validateCommitActionsWithLookup(context.Background(), actions, func(ctx context.Context, name string) (int, error) {
		if name == "Jenny" {
			return 7, nil
		}
		return 0, nil
	}, func(ctx context.Context, id int) (bool, error) {
		return false, nil
	})
	if err == nil {
		t.Fatal("expected existing new_identity name to be rejected")
	}
	if !strings.Contains(err.Error(), "use 'merge' or 'new_variant' instead") {
		t.Fatalf("expected reviewer guidance in error, got: %v", err)
	}
}

func TestValidateCommitActionsWithLookupAllowsFreshNewIdentityName(t *testing.T) {
	actions := []store.CommitAction{
		{
			TrackID:      "1",
			Action:       "new_identity",
			IdentityName: "BrandNewPerson",
		},
	}

	err := validateCommitActionsWithLookup(context.Background(), actions, func(ctx context.Context, name string) (int, error) {
		return 0, nil
	}, func(ctx context.Context, id int) (bool, error) {
		return false, nil
	})
	if err != nil {
		t.Fatalf("expected fresh new_identity name to validate, got: %v", err)
	}
}

func TestValidateCommitActionsWithLookupRejectsDuplicateNewIdentityNamesWithinBatch(t *testing.T) {
	actions := []store.CommitAction{
		{
			TrackID:      "1",
			Action:       "new_identity",
			IdentityName: "Jenny",
		},
		{
			TrackID:      "2",
			Action:       "new_identity",
			IdentityName: "Jenny",
		},
	}

	err := validateCommitActionsWithLookup(context.Background(), actions, func(ctx context.Context, name string) (int, error) {
		return 0, nil
	}, func(ctx context.Context, id int) (bool, error) {
		return false, nil
	})
	if err == nil {
		t.Fatal("expected duplicate new_identity names in the same batch to be rejected")
	}
	if !strings.Contains(err.Error(), "duplicated in this review batch") {
		t.Fatalf("expected duplicate-batch guidance in error, got: %v", err)
	}
}

func TestValidateCommitActionsWithLookupRejectsCaseInsensitiveDuplicateNewIdentityNamesWithinBatch(t *testing.T) {
	actions := []store.CommitAction{
		{
			TrackID:      "1",
			Action:       "new_identity",
			IdentityName: "Jenny",
		},
		{
			TrackID:      "2",
			Action:       "new_identity",
			IdentityName: "jEnNy",
		},
	}

	err := validateCommitActionsWithLookup(context.Background(), actions, func(ctx context.Context, name string) (int, error) {
		return 0, nil
	}, func(ctx context.Context, id int) (bool, error) {
		return false, nil
	})
	if err == nil {
		t.Fatal("expected case-insensitive duplicate new_identity names in the same batch to be rejected")
	}
	if !strings.Contains(err.Error(), "duplicated in this review batch") {
		t.Fatalf("expected duplicate-batch guidance in error, got: %v", err)
	}
}

func TestValidateCommitActionsWithLookupRejectsSystemLabelCollision(t *testing.T) {
	actions := []store.CommitAction{
		{
			TrackID:      "1",
			Action:       "new_identity",
			IdentityName: "Identity 42",
		},
	}

	err := validateCommitActionsWithLookup(context.Background(), actions, func(ctx context.Context, name string) (int, error) {
		return 0, nil
	}, func(ctx context.Context, id int) (bool, error) {
		return id == 42, nil
	})
	if err == nil {
		t.Fatal("expected system label collision to be rejected")
	}
	if !strings.Contains(err.Error(), "conflicts with Sentinel's system label") {
		t.Fatalf("expected system-label guidance in error, got: %v", err)
	}
}

func TestValidateCommitActionsWithLookupRejectsCaseInsensitiveSystemLabelCollision(t *testing.T) {
	actions := []store.CommitAction{
		{
			TrackID:      "1",
			Action:       "new_identity",
			IdentityName: "identity 42",
		},
	}

	err := validateCommitActionsWithLookup(context.Background(), actions, func(ctx context.Context, name string) (int, error) {
		return 0, nil
	}, func(ctx context.Context, id int) (bool, error) {
		return id == 42, nil
	})
	if err == nil {
		t.Fatal("expected case-insensitive system label collision to be rejected")
	}
	if !strings.Contains(err.Error(), "conflicts with Sentinel's system label") {
		t.Fatalf("expected system-label guidance in error, got: %v", err)
	}
}

func TestValidateCommitActionsWithLookupAllowsNonExactIdentityPrefixName(t *testing.T) {
	actions := []store.CommitAction{
		{
			TrackID:      "1",
			Action:       "new_identity",
			IdentityName: "Identity 42 - Lobby",
		},
	}

	err := validateCommitActionsWithLookup(context.Background(), actions, func(ctx context.Context, name string) (int, error) {
		return 0, nil
	}, func(ctx context.Context, id int) (bool, error) {
		return true, nil
	})
	if err != nil {
		t.Fatalf("expected non-exact system-label-like name to validate, got: %v", err)
	}
}
