package cmd

import (
	"strings"
	"testing"
)

func TestReadReviewDocumentTreatsTracksMapAsCurrentFormat(t *testing.T) {
	data := []byte(`
tracks:
  - track_id: Track_1
    action: merge
`)

	review, legacyFormat, err := readReviewDocument(data)
	if err != nil {
		t.Fatalf("readReviewDocument returned error: %v", err)
	}
	if legacyFormat {
		t.Fatal("expected review document with tracks key to be treated as current format")
	}
	if len(review.Tracks) != 1 || review.Tracks[0].TrackID != "Track_1" {
		t.Fatalf("unexpected review tracks: %+v", review.Tracks)
	}
}

func TestPrepareCommitBatchRejectsBlankActions(t *testing.T) {
	review := ReviewDocument{
		VideoID:   "vid_test",
		InputPath: "/tmp/test.mp4",
		Tracks: []StagingItem{
			{TrackID: "Track_1", Action: ""},
			{TrackID: "Track_2", Action: "discard"},
		},
	}

	_, _, _, err := prepareCommitBatch(review, false)
	if err == nil {
		t.Fatal("expected blank action to be rejected")
	}
	if !strings.Contains(err.Error(), "Track_1") {
		t.Fatalf("expected unresolved track ID in error, got: %v", err)
	}
}

func TestPrepareCommitBatchRejectsCurrentFormatWithoutVideoMetadata(t *testing.T) {
	review := ReviewDocument{
		Tracks: []StagingItem{
			{TrackID: "Track_1", Action: "discard"},
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
				TrackID:        "Track_1",
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
