package cmd

import (
	"context"
	"fmt"
	"math"
	"os"
	"testing"
	"time"

	"github.com/andresmejia3/sentinel/internal/store"
	"github.com/andresmejia3/sentinel/internal/types"
	"github.com/andresmejia3/sentinel/internal/utils"
	"github.com/testcontainers/testcontainers-go"
	"github.com/testcontainers/testcontainers-go/modules/postgres"
	"github.com/testcontainers/testcontainers-go/wait"
)

type scanDBStub struct {
	matchVariantID   int
	matchIdentityID  int
	matchIdentity    string
	matchVariantName string
	matchErr         error
}

func (s scanDBStub) FindClosestIdentity(ctx context.Context, vec []float64, threshold float64) (int, int, string, string, error) {
	return s.matchVariantID, s.matchIdentityID, s.matchIdentity, s.matchVariantName, s.matchErr
}

func (s scanDBStub) FindTopIdentities(ctx context.Context, vec []float64, limit int) ([]store.IdentityMatch, error) {
	return nil, nil
}

func (s scanDBStub) CreateIdentity(ctx context.Context, vec []float64, count int) (int, int, error) {
	return 0, 0, fmt.Errorf("unexpected CreateIdentity call in test")
}

func (s scanDBStub) UpdateIdentity(ctx context.Context, id int, newVec []float64, newCount int) error {
	return fmt.Errorf("unexpected UpdateIdentity call in test")
}

func TestCosineDist(t *testing.T) {
	tests := []struct {
		name string
		a    []float64 // Must be normalized (length 1.0) as per function contract
		b    []float64
		want float64
	}{
		{
			name: "Identical vectors",
			a:    []float64{1.0, 0.0},
			b:    []float64{1.0, 0.0},
			want: 0.0, // 1 - (1 / 1)
		},
		{
			name: "Orthogonal vectors",
			a:    []float64{1.0, 0.0},
			b:    []float64{0.0, 1.0},
			want: 1.0, // 1 - (0 / 1)
		},
		{
			name: "Opposite vectors",
			a:    []float64{1.0, 0.0},
			b:    []float64{-1.0, 0.0},
			want: 2.0, // 1 - (-1 / 1)
		},
		{
			name: "B is unnormalized (scaled)",
			a:    []float64{1.0, 0.0},
			b:    []float64{5.0, 0.0}, // Length 5
			want: 0.0,                 // 1 - (5 / 5) = 0. Direction is same.
		},
		{
			name: "Empty vectors",
			a:    []float64{},
			b:    []float64{},
			want: 1.0, // Safety fallback
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := utils.CosineDist(tt.a, tt.b)
			// Use epsilon for float comparison
			if math.Abs(got-tt.want) > 1e-9 {
				t.Errorf("CosineDist() = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestScanPersistence simulates the database logic inside runScan's aggregator.
// It verifies that tracks are correctly persisted to the DB.
func TestScanPersistence(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	ctx := context.Background()

	// Explicitly check for Docker availability and fail hard if missing
	// We wrap this in a function to recover from panics inside testcontainers (e.g. socket not found)
	err := func() (err error) {
		defer func() {
			if r := recover(); r != nil {
				err = fmt.Errorf("testcontainers panicked: %v", r)
			}
		}()
		_, err = testcontainers.NewDockerClientWithOpts(ctx)
		return
	}()
	if err != nil {
		t.Skipf("Docker not available, skipping integration test: %v", err)
	}

	// Start Postgres Container
	pgContainer, err := postgres.Run(ctx,
		"pgvector/pgvector:pg16",
		postgres.WithDatabase("sentinel_test"),
		postgres.WithUsername("user"),
		postgres.WithPassword("password"),
		testcontainers.WithWaitStrategy(
			wait.ForLog("database system is ready to accept connections").
				WithOccurrence(2).
				WithStartupTimeout(5*time.Second)),
		testcontainers.WithLogger(noopLogger{}),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer pgContainer.Terminate(ctx)

	connStr, _ := pgContainer.ConnectionString(ctx, "sslmode=disable")
	db, err := store.New(ctx, connStr)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close(ctx)

	// Setup Test Data
	videoID := "vid_test_123"
	db.EnsureVideoMetadata(ctx, videoID, "/tmp/test.mp4")

	// Simulate a Track Persistence
	// This mirrors the logic inside the persistTrack closure in scan.go
	vec := make([]float64, 512)
	vec[0] = 1.0

	variantID, masterID, err := db.CreateIdentity(ctx, vec, 10)
	if err != nil {
		t.Fatalf("Failed to create identity: %v", err)
	}

	// Update Identity (Simulate finding the person again)
	// Old: [1.0, 0.0...], Count: 10
	// New: [0.0, 1.0...], Count: 10
	// Expected Avg: [0.5, 0.5...], Total Count: 20
	vecUpdate := make([]float64, 512)
	vecUpdate[1] = 1.0
	if err := db.UpdateIdentity(ctx, variantID, vecUpdate, 10); err != nil {
		t.Fatalf("Failed to update identity: %v", err)
	}

	if err = db.InsertInterval(ctx, videoID, 0.0, 5.0, 10, variantID); err != nil {
		t.Fatalf("Failed to insert interval: %v", err)
	}

	// Verify Vector Math
	variants, err := db.GetVariantsForIdentities(ctx, []int{masterID})
	if err != nil {
		t.Fatalf("Failed to retrieve vectors: %v", err)
	}
	if len(variants) == 0 || variants[0].VariantID != variantID {
		t.Fatalf("Expected variant %d, got %v", variantID, variants)
	}
	gotVec := variants[0].Vec
	if math.Abs(gotVec[0]-0.5) > 1e-5 || math.Abs(gotVec[1]-0.5) > 1e-5 {
		t.Errorf("UpdateIdentity failed weighted average. Expected ~0.5 at indices 0&1, got %v", gotVec[:2])
	}

	// Verify Intervals
	intervals, err := db.GetIntervalsForIdentity(ctx, masterID)
	if err != nil {
		t.Fatalf("Failed to get intervals for verification: %v", err)
	}
	if len(intervals) != 1 {
		t.Fatalf("Expected 1 interval for identity %d, got %d", masterID, len(intervals))
	}
	if intervals[0].VideoID != videoID || intervals[0].Start != 0.0 || intervals[0].End != 5.0 {
		t.Errorf("Mismatch in persisted interval data. Got %+v", intervals[0])
	}
}

func TestValidateScanFlags(t *testing.T) {
	// Create a temp file for valid input
	tmpFile, err := os.CreateTemp("", "video.mp4")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpFile.Name())
	tmpFile.Close()

	// Create a temp dir for invalid input
	tmpDir, err := os.MkdirTemp("", "testdir")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	tests := []struct {
		name    string
		opts    Options
		wantErr bool
	}{
		{
			name: "Valid options",
			opts: Options{
				InputPath:      tmpFile.Name(),
				NthFrame:       1,
				MatchThreshold: 0.5,
				BlipDuration:   "100ms",
				GracePeriod:    "1s",
				WorkerTimeout:  "30s",
			},
			wantErr: false,
		},
		{
			name: "Input file does not exist",
			opts: Options{
				InputPath: "nonexistent.mp4",
			},
			wantErr: true,
		},
		{
			name: "Input is directory",
			opts: Options{
				InputPath: tmpDir,
			},
			wantErr: true,
		},
		{
			name: "Invalid NthFrame",
			opts: Options{
				InputPath: tmpFile.Name(),
				NthFrame:  0,
			},
			wantErr: true,
		},
		{
			name: "Invalid MatchThreshold",
			opts: Options{
				InputPath:      tmpFile.Name(),
				NthFrame:       1,
				MatchThreshold: 1.5,
			},
			wantErr: true,
		},
		{
			name: "NoStaging cannot be combined with review file",
			opts: Options{
				InputPath:      tmpFile.Name(),
				NthFrame:       1,
				MatchThreshold: 0.5,
				BlipDuration:   "100ms",
				GracePeriod:    "1s",
				WorkerTimeout:  "30s",
				NoStaging:      true,
				ReviewFile:     "scan.review.yaml",
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Redirect stderr to discard output during this specific sub-test
			oldStderr := os.Stderr
			r, w, _ := os.Pipe()
			os.Stderr = w

			if err := validateScanFlags(&tt.opts); (err != nil) != tt.wantErr {
				t.Errorf("validateScanFlags() error = %v, wantErr %v", err, tt.wantErr)
			}

			// Restore stderr and close the pipe
			w.Close()
			os.Stderr = oldStderr
			r.Close()
		})
	}
}

func TestAggregateFrameResultsPrunesExpiredTracksBeforeMatching(t *testing.T) {
	vec := make([]float64, embeddingDim)
	vec[0] = 1.0

	stale := newActiveTrack(42, 42, 0, vec, []byte("thumb"), 0.9, "Identity 42", "Default", true, []int{0, 0, 10, 10})
	stale.LastFrame = 0
	tracks := []*activeTrack{stale}

	var persisted []*activeTrack
	idNames := map[int]identityNameData{42: {IdentityName: "Identity 42", VariantName: "Default"}}
	variantToIdentityID := map[int]int{42: 42}
	tempIDCounter := -1

	frame := scanResult{
		Index: 25,
		Faces: []types.FaceResult{{
			Loc:     []int{0, 0, 10, 10},
			Vec:     vec,
			Thumb:   []byte("new"),
			Quality: 0.8,
		}},
	}

	aggregateFrameResults(
		context.Background(),
		frame,
		&tracks,
		new(int),
		Options{MatchThreshold: 0.6},
		scanDBStub{matchVariantID: -1},
		make(chan error, 1),
		variantToIdentityID,
		idNames,
		&tempIDCounter,
		10,
		func(t *activeTrack) { persisted = append(persisted, t) },
	)

	if len(persisted) != 1 || persisted[0].ID != 42 {
		t.Fatalf("expected stale track 42 to be persisted before matching, got %+v", persisted)
	}
	if len(tracks) != 1 {
		t.Fatalf("expected exactly one active track after processing, got %d", len(tracks))
	}
	if tracks[0].ID >= 0 {
		t.Fatalf("expected reappearing face to start a new pending track, got ID %d", tracks[0].ID)
	}
}

func TestAggregateFrameResultsDoesNotDropFaceWhenBestDBMatchIsAlreadyActive(t *testing.T) {
	activeVec := make([]float64, embeddingDim)
	activeVec[0] = 1.0

	existingTrack := newActiveTrack(7, 7, 0, activeVec, []byte("thumb"), 0.9, "Identity 7", "Default", true, []int{0, 0, 10, 10})
	tracks := []*activeTrack{existingTrack}

	idNames := map[int]identityNameData{7: {IdentityName: "Identity 7", VariantName: "Default"}}
	variantToIdentityID := map[int]int{7: 7}
	tempIDCounter := -1
	totalDetections := 0

	newFaceVec := make([]float64, embeddingDim)
	newFaceVec[1] = 1.0

	frame := scanResult{
		Index: 1,
		Faces: []types.FaceResult{{
			Loc:     []int{20, 20, 40, 40},
			Vec:     newFaceVec,
			Thumb:   []byte("new"),
			Quality: 0.7,
		}},
	}

	aggregateFrameResults(
		context.Background(),
		frame,
		&tracks,
		&totalDetections,
		Options{MatchThreshold: 0.6},
		scanDBStub{
			matchVariantID:   7,
			matchIdentityID:  7,
			matchIdentity:    "Identity 7",
			matchVariantName: "Default",
		},
		make(chan error, 1),
		variantToIdentityID,
		idNames,
		&tempIDCounter,
		10,
		func(*activeTrack) {},
	)

	if totalDetections != 1 {
		t.Fatalf("expected one detection to be processed, got %d", totalDetections)
	}
	if len(tracks) != 2 {
		t.Fatalf("expected active track plus a new pending track, got %d track(s)", len(tracks))
	}
	if tracks[1].ID >= 0 {
		t.Fatalf("expected second face to become a pending track, got ID %d", tracks[1].ID)
	}
}

func TestAggregateFrameResultsResolvesTrackConflictsWithoutFaceOrderBias(t *testing.T) {
	trackVec := make([]float64, embeddingDim)
	trackVec[0] = 1.0

	existingTrack := newActiveTrack(11, 11, 0, trackVec, []byte("existing"), 0.8, "Identity 11", "Default", true, []int{0, 0, 10, 10})
	tracks := []*activeTrack{existingTrack}

	idNames := map[int]identityNameData{11: {IdentityName: "Identity 11", VariantName: "Default"}}
	variantToIdentityID := map[int]int{11: 11}
	tempIDCounter := -1
	totalDetections := 0

	weakerVec := make([]float64, embeddingDim)
	weakerVec[0] = 0.8
	weakerVec[1] = 0.6

	strongerVec := make([]float64, embeddingDim)
	strongerVec[0] = 1.0

	frame := scanResult{
		Index: 1,
		Faces: []types.FaceResult{
			{
				Loc:     []int{20, 20, 40, 40},
				Vec:     weakerVec,
				Thumb:   []byte("weaker"),
				Quality: 0.7,
			},
			{
				Loc:     []int{50, 50, 70, 70},
				Vec:     strongerVec,
				Thumb:   []byte("stronger"),
				Quality: 0.95,
			},
		},
	}

	aggregateFrameResults(
		context.Background(),
		frame,
		&tracks,
		&totalDetections,
		Options{MatchThreshold: 0.6},
		scanDBStub{matchVariantID: -1},
		make(chan error, 1),
		variantToIdentityID,
		idNames,
		&tempIDCounter,
		10,
		func(*activeTrack) {},
	)

	if totalDetections != 2 {
		t.Fatalf("expected two detections to be processed, got %d", totalDetections)
	}
	if len(tracks) != 2 {
		t.Fatalf("expected one resolved track and one pending conflict track, got %d tracks", len(tracks))
	}
	if tracks[0].ID != 11 {
		t.Fatalf("expected existing track to keep ID 11, got %d", tracks[0].ID)
	}
	if tracks[0].LastScore != 0.95 {
		t.Fatalf("expected strongest claimant to win the active track, got last score %.2f", tracks[0].LastScore)
	}
	if tracks[1].ID >= 0 {
		t.Fatalf("expected weaker conflicting face to become a pending track, got ID %d", tracks[1].ID)
	}
	if tracks[1].LastScore != 0.7 {
		t.Fatalf("expected pending track to preserve weaker conflicting face, got last score %.2f", tracks[1].LastScore)
	}
}

type noopLogger struct{}

func (n noopLogger) Printf(format string, v ...interface{}) {}
