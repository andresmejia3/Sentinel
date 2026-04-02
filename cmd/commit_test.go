package cmd

import (
	"context"
	"strings"
	"testing"

	"github.com/andresmejia3/sentinel/internal/store"
)

func TestReadReviewDocumentTreatsTracksMapAsCurrentFormat(t *testing.T) {
	data := []byte(`
tracks:
  - id: 1
    action: merge
`)

	review, err := readReviewDocument(data)
	if err != nil {
		t.Fatalf("readReviewDocument returned error: %v", err)
	}
	if len(review.Tracks) != 1 || review.Tracks[0].ID != 1 {
		t.Fatalf("unexpected review tracks: %+v", review.Tracks)
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
	if !strings.Contains(err.Error(), "current review document format") {
		t.Fatalf("expected current-format guidance in error, got: %v", err)
	}
}

func TestReadReviewDocumentRejectsEmbeddedInternalDataFields(t *testing.T) {
	data := []byte(`
review_id: a1b2c3d4e5f6
video_id: vid_test
input_path: samples/test.mp4
tracks:
  - id: 1
    action: merge
    internal_vector: [1, 2, 3]
`)

	_, err := readReviewDocument(data)
	if err == nil {
		t.Fatal("expected embedded internal data fields to be rejected")
	}
	if !strings.Contains(err.Error(), "internal_vector") {
		t.Fatalf("expected unknown internal_vector field in error, got: %v", err)
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
