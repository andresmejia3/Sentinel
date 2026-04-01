package cmd

import (
	"strings"
	"testing"
)

func TestReadReviewDocumentTreatsTracksMapAsCurrentFormat(t *testing.T) {
	data := []byte(`
tracks:
  - id: 1
    action: merge
`)

	review, legacyFormat, err := readReviewDocument(data)
	if err != nil {
		t.Fatalf("readReviewDocument returned error: %v", err)
	}
	if legacyFormat {
		t.Fatal("expected review document with tracks key to be treated as current format")
	}
	if len(review.Tracks) != 1 || review.Tracks[0].ID != 1 {
		t.Fatalf("unexpected review tracks: %+v", review.Tracks)
	}
}

func TestPrepareCommitBatchRejectsBlankActions(t *testing.T) {
	review := ReviewDocument{
		VideoID:   "vid_test",
		InputPath: "/tmp/test.mp4",
		Tracks: []StagingItem{
			{ID: 1, Action: ""},
			{ID: 2, Action: "discard"},
		},
	}

	_, _, _, err := prepareCommitBatch(review, false)
	if err == nil {
		t.Fatal("expected blank action to be rejected")
	}
	if !strings.Contains(err.Error(), "1") {
		t.Fatalf("expected unresolved track ID in error, got: %v", err)
	}
}

func TestPrepareCommitBatchRejectsDuplicateReviewIDs(t *testing.T) {
	review := ReviewDocument{
		VideoID:   "vid_test",
		InputPath: "/tmp/test.mp4",
		Tracks: []StagingItem{
			{ID: 1, Action: "discard"},
			{ID: 1, Action: "discard"},
		},
	}

	_, _, _, err := prepareCommitBatch(review, false)
	if err == nil {
		t.Fatal("expected duplicate review IDs to be rejected")
	}
	if !strings.Contains(err.Error(), "duplicate track id '1'") {
		t.Fatalf("expected duplicate review ID error, got: %v", err)
	}
}

func TestPrepareCommitBatchRejectsCurrentFormatWithoutVideoMetadata(t *testing.T) {
	review := ReviewDocument{
		Tracks: []StagingItem{
			{ID: 1, Action: "discard"},
		},
	}

	_, _, _, err := prepareCommitBatch(review, false)
	if err == nil {
		t.Fatal("expected missing video metadata to be rejected")
	}
	if !strings.Contains(err.Error(), "video_id") {
		t.Fatalf("expected missing video_id error, got: %v", err)
	}
}

func TestPrepareCommitBatchAllowsLegacyReviewWithoutVideoMetadata(t *testing.T) {
	vec := make([]float64, 512)
	vec[0] = 1.0

	review := ReviewDocument{
		Tracks: []StagingItem{
			{
				ID:             1,
				Action:         "new_identity",
				InternalVector: vec,
				InternalCount:  3,
			},
		},
	}

	actions, intervals, skipCount, err := prepareCommitBatch(review, true)
	if err != nil {
		t.Fatalf("expected legacy review to validate, got error: %v", err)
	}
	if len(actions) != 1 {
		t.Fatalf("expected one action, got %d", len(actions))
	}
	if len(intervals) != 0 {
		t.Fatalf("expected legacy review to omit intervals, got %d", len(intervals))
	}
	if skipCount != 0 {
		t.Fatalf("expected zero skipped items, got %d", skipCount)
	}
}

func TestPrepareCommitBatchUsesGroundTruthIdentityFields(t *testing.T) {
	vec := make([]float64, 512)
	vec[0] = 1.0

	review := ReviewDocument{
		VideoID:   "vid_test",
		InputPath: "/tmp/test.mp4",
		Tracks: []StagingItem{
			{
				ID:                      1,
				Action:                  "new_variant",
				Identity:                "Jenny",
				Variant:                 "Side_Profile",
				LegacySuggestedIdentity: "Wrong",
				LegacySuggestedVariant:  "WrongVariant",
				InternalVector:          vec,
				InternalCount:           2,
			},
		},
	}

	actions, _, _, err := prepareCommitBatch(review, false)
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

func TestPrepareCommitBatchRejectsNewVariantWithoutVariantName(t *testing.T) {
	vec := make([]float64, 512)
	vec[0] = 1.0

	review := ReviewDocument{
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

	_, _, _, err := prepareCommitBatch(review, false)
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

	_, _, _, err := prepareCommitBatch(review, false)
	if err == nil {
		t.Fatal("expected new_identity with explicit variant to be rejected")
	}
	if !strings.Contains(err.Error(), "requires `variant` to be blank") {
		t.Fatalf("expected variant-blank validation error, got: %v", err)
	}
}
