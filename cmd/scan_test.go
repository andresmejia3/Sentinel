package cmd

import (
	"context"
	"fmt"
	"math"
	"os"
	"testing"
	"time"

	"github.com/andresmejia3/sentinel/internal/store"
	"github.com/andresmejia3/sentinel/internal/utils"
	"github.com/testcontainers/testcontainers-go"
	"github.com/testcontainers/testcontainers-go/modules/postgres"
	"github.com/testcontainers/testcontainers-go/wait"
)

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
		t.Fatalf("Docker not available, cannot run integration test: %v", err)
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

	id, err := db.CreateIdentity(ctx, vec, 10)
	if err != nil {
		t.Fatalf("Failed to create identity: %v", err)
	}

	// Update Identity (Simulate finding the person again)
	// Old: [1.0, 0.0...], Count: 10
	// New: [0.0, 1.0...], Count: 10
	// Expected Avg: [0.5, 0.5...], Total Count: 20
	vecUpdate := make([]float64, 512)
	vecUpdate[1] = 1.0
	if err := db.UpdateIdentity(ctx, id, vecUpdate, 10); err != nil {
		t.Fatalf("Failed to update identity: %v", err)
	}

	if err = db.InsertInterval(ctx, videoID, 0.0, 5.0, 10, id); err != nil {
		t.Fatalf("Failed to insert interval: %v", err)
	}

	// Verify Vector Math
	vecs, err := db.GetIdentityVectors(ctx, []int{id})
	if err != nil {
		t.Fatalf("Failed to retrieve vectors: %v", err)
	}
	gotVec := vecs[id]
	if math.Abs(gotVec[0]-0.5) > 1e-5 || math.Abs(gotVec[1]-0.5) > 1e-5 {
		t.Errorf("UpdateIdentity failed weighted average. Expected ~0.5 at indices 0&1, got %v", gotVec[:2])
	}

	// Verify Intervals
	intervals, err := db.GetIdentityIntervals(ctx, id)
	if err != nil {
		t.Fatalf("Failed to get intervals for verification: %v", err)
	}
	if len(intervals) != 1 {
		t.Fatalf("Expected 1 interval for identity %d, got %d", id, len(intervals))
	}
	if intervals[0].VideoID != videoID || intervals[0].Start != 0.0 || intervals[0].End != 5.0 {
		t.Errorf("Mismatch in persisted interval data. Got %+v", intervals[0])
	}
}

func TestFmtTime(t *testing.T) {
	tests := []struct {
		seconds float64
		want    string
	}{
		{0, "00:00:00"},
		{65, "00:01:05"},
		{3661, "01:01:01"},
	}

	for _, tt := range tests {
		if got := fmtTime(tt.seconds); got != tt.want {
			t.Errorf("fmtTime(%v) = %v, want %v", tt.seconds, got, tt.want)
		}
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
				InputPath:       tmpFile.Name(),
				NthFrame:        1,
				MatchThreshold:  0.5,
				GracePeriod:     "1s",
				WorkerTimeout:   "30s",
				QualityStrategy: "clarity",
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

type noopLogger struct{}

func (n noopLogger) Printf(format string, v ...interface{}) {}
