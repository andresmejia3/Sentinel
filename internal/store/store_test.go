package store

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/testcontainers/testcontainers-go"
	"github.com/testcontainers/testcontainers-go/modules/postgres"
	"github.com/testcontainers/testcontainers-go/wait"
)

// TestStoreIntegration runs a full integration test against a real Postgres container.
// It requires Docker to be running.
func TestStoreIntegration(t *testing.T) {
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

	// Start Postgres Container with pgvector
	// We use the official pgvector image to ensure the extension is available.
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
		if isDockerUnavailable(err) {
			t.Skipf("Docker not available, skipping integration test: %v", err)
		}
		t.Fatalf("Failed to start postgres container: %v", err)
	}
	defer func() {
		if err := pgContainer.Terminate(ctx); err != nil {
			t.Fatalf("Failed to terminate container: %v", err)
		}
	}()

	// Get Connection String
	connStr, err := pgContainer.ConnectionString(ctx, "sslmode=disable")
	if err != nil {
		t.Fatalf("Failed to get connection string: %v", err)
	}

	// Initialize Store (runs migrations)
	s, err := New(ctx, connStr)
	if err != nil {
		t.Fatalf("Failed to connect to store: %v", err)
	}
	defer s.Close(ctx)

	// --- Test Scenarios ---

	vecA := make([]float64, 512)
	vecA[0] = 1.0 // Vector A points along X axis
	idA, _, err := s.CreateIdentity(ctx, vecA, 1)
	if err != nil {
		t.Fatalf("CreateIdentity failed: %v", err)
	}
	if idA <= 0 {
		t.Errorf("Expected positive ID, got %d", idA)
	}

	// Find Closest Identity (Exact Match)
	matchID, _, _, _, err := s.FindClosestIdentity(ctx, vecA, 0.1)
	if err != nil {
		t.Fatalf("FindClosestIdentity failed: %v", err)
	}
	if matchID != idA {
		t.Errorf("Expected match ID %d, got %d", idA, matchID)
	}

	// Find Closest Identity (No Match)
	vecB := make([]float64, 512)
	vecB[1] = 1.0 // Vector B points along Y axis (Orthogonal to A)
	// Distance should be ~1.0 (Cosine Dist). Threshold 0.1 should fail.
	noMatchID, _, _, _, err := s.FindClosestIdentity(ctx, vecB, 0.1)
	if err != nil {
		t.Fatalf("FindClosestIdentity error: %v", err)
	}
	if noMatchID != -1 {
		t.Errorf("Expected no match (-1), got %d", noMatchID)
	}

	// Commit New Identity With Explicit Name
	vecNamed := make([]float64, 512)
	vecNamed[2] = 1.0
	if err := s.ApplyCommitBatch(ctx, "commit_named_identity", []CommitAction{{
		TrackID:      "Track_named",
		Action:       "new_identity",
		IdentityName: "Identity 42 - Lobby",
		Vector:       vecNamed,
		Count:        1,
	}}, "", "", nil); err != nil {
		t.Fatalf("ApplyCommitBatch(commit_named_identity) failed: %v", err)
	}

	var namedIdentityCount int
	if err := s.pool.QueryRow(ctx, "SELECT COUNT(*) FROM identities WHERE name = $1", "Identity 42 - Lobby").Scan(&namedIdentityCount); err != nil {
		t.Fatalf("failed to count explicitly named identity: %v", err)
	}
	if namedIdentityCount != 1 {
		t.Fatalf("expected explicitly named identity to be preserved, got count=%d", namedIdentityCount)
	}

	caseInsensitiveIdentityID, err := s.GetIdentityIDByName(ctx, "identity 42 - lobby")
	if err != nil {
		t.Fatalf("GetIdentityIDByName(case-insensitive) failed: %v", err)
	}
	if caseInsensitiveIdentityID == 0 {
		t.Fatal("expected case-insensitive identity lookup to find explicitly named identity")
	}

	glassesVec := make([]float64, 512)
	glassesVec[3] = 1.0
	glassesVariantID, err := s.CreateVariant(ctx, caseInsensitiveIdentityID, glassesVec, 1, "Glasses")
	if err != nil {
		t.Fatalf("CreateVariant failed: %v", err)
	}
	if glassesVariantID <= 0 {
		t.Fatalf("expected positive glasses variant ID, got %d", glassesVariantID)
	}

	caseInsensitiveVariantID, err := s.GetVariantID(ctx, caseInsensitiveIdentityID, "glasses")
	if err != nil {
		t.Fatalf("GetVariantID(case-insensitive) failed: %v", err)
	}
	if caseInsensitiveVariantID != glassesVariantID {
		t.Fatalf("expected case-insensitive variant lookup to find %d, got %d", glassesVariantID, caseInsensitiveVariantID)
	}

	_, err = s.CreateVariant(ctx, caseInsensitiveIdentityID, glassesVec, 1, "gLaSsEs")
	if err == nil {
		t.Fatal("expected case-insensitive duplicate variant name to be rejected")
	}
	if !strings.Contains(strings.ToLower(err.Error()), "duplicate") {
		t.Fatalf("expected duplicate variant error, got: %v", err)
	}

	// Update Identity (Weighted Average)
	// We update ID A with a new vector that is slightly different.
	// Old: [1.0, 0.0...] (Count 1)
	// New: [0.0, 1.0...] (Count 1)
	// Avg: [0.5, 0.5...] (Count 2)
	err = s.UpdateIdentity(ctx, idA, vecB, 1)
	if err != nil {
		t.Fatalf("UpdateIdentity failed: %v", err)
	}

	// Verify the update directly by fetching the vector
	// idA is a Variant ID. We need to find its Master ID to use GetVariantsForIdentities,
	// or just check the DB directly. For this test, let's assume we know the Master ID is 1 (since it's the first).
	// A better way is to check the variant update logic.
	// Let's use FindClosestIdentity to verify the vector has moved.
	// After averaging vecA and vecB, the stored embedding is no longer an
	// exact match for vecB, so the threshold needs to allow that expected drift.
	matchID, _, _, _, err = s.FindClosestIdentity(ctx, vecB, 0.3)
	if err != nil {
		t.Fatalf("FindClosestIdentity failed after update: %v", err)
	}
	if matchID != idA {
		t.Errorf("Expected to find updated variant %d, got %d", idA, matchID)
	}

	err = s.EnsureVideoMetadata(ctx, "vid_123", "/tmp/video.mp4")
	if err != nil {
		t.Fatalf("EnsureVideoMetadata failed: %v", err)
	}
	err = s.InsertInterval(ctx, "vid_123", 0.0, 5.0, 10, idA)
	if err != nil {
		t.Fatalf("InsertInterval failed: %v", err)
	}

	identities, err := s.ListIdentities(ctx, 0, 0, "")
	if err != nil {
		t.Fatalf("ListIdentities failed: %v", err)
	}
	if len(identities) != 1 {
		t.Errorf("Expected 1 identity, got %d", len(identities))
	}
	if identities[0].Count != 2 {
		t.Errorf("Expected count 2 (1 initial + 1 update), got %d", identities[0].Count)
	}
}

func TestRevertCommitRestoresVideoState(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	ctx := context.Background()

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
		if isDockerUnavailable(err) {
			t.Skipf("Docker not available, skipping integration test: %v", err)
		}
		t.Fatalf("Failed to start postgres container: %v", err)
	}
	defer func() {
		if err := pgContainer.Terminate(ctx); err != nil {
			t.Fatalf("Failed to terminate container: %v", err)
		}
	}()

	connStr, err := pgContainer.ConnectionString(ctx, "sslmode=disable")
	if err != nil {
		t.Fatalf("Failed to get connection string: %v", err)
	}

	s, err := New(ctx, connStr)
	if err != nil {
		t.Fatalf("Failed to connect to store: %v", err)
	}
	defer s.Close(ctx)

	originalVec := make([]float64, 512)
	originalVec[0] = 1.0
	originalVariantID, _, err := s.CreateIdentity(ctx, originalVec, 3)
	if err != nil {
		t.Fatalf("CreateIdentity failed: %v", err)
	}

	const videoID = "vid_restore"
	if err := s.EnsureVideoMetadata(ctx, videoID, "/tmp/original.mp4"); err != nil {
		t.Fatalf("EnsureVideoMetadata failed: %v", err)
	}
	if err := s.InsertInterval(ctx, videoID, 1.0, 2.5, 3, originalVariantID); err != nil {
		t.Fatalf("InsertInterval failed: %v", err)
	}

	newVec := make([]float64, 512)
	newVec[1] = 1.0
	commitID := "commit_restore_test"
	actions := []CommitAction{
		{
			TrackID: "Track_1",
			Action:  "new_identity",
			Vector:  newVec,
			Count:   4,
		},
	}
	intervals := []CommitInterval{
		{
			TrackID:   "Track_1",
			Start:     10.0,
			End:       12.0,
			FaceCount: 4,
		},
	}

	if err := s.ApplyCommitBatch(ctx, commitID, actions, videoID, "/tmp/updated.mp4", intervals); err != nil {
		t.Fatalf("ApplyCommitBatch failed: %v", err)
	}

	var committedPath string
	if err := s.pool.QueryRow(ctx, "SELECT path FROM video_metadata WHERE id = $1", videoID).Scan(&committedPath); err != nil {
		t.Fatalf("failed to read committed metadata: %v", err)
	}
	if committedPath != "/tmp/updated.mp4" {
		t.Fatalf("video path after commit = %q, want %q", committedPath, "/tmp/updated.mp4")
	}

	var committedIntervals int
	if err := s.pool.QueryRow(ctx, "SELECT COUNT(*) FROM face_intervals WHERE video_id = $1", videoID).Scan(&committedIntervals); err != nil {
		t.Fatalf("failed to count committed intervals: %v", err)
	}
	if committedIntervals != 1 {
		t.Fatalf("expected 1 committed interval, got %d", committedIntervals)
	}

	if err := s.RevertCommit(ctx, commitID); err != nil {
		t.Fatalf("RevertCommit failed: %v", err)
	}

	var restoredPath string
	if err := s.pool.QueryRow(ctx, "SELECT path FROM video_metadata WHERE id = $1", videoID).Scan(&restoredPath); err != nil {
		t.Fatalf("failed to read restored metadata: %v", err)
	}
	if restoredPath != "/tmp/original.mp4" {
		t.Fatalf("video path after rollback = %q, want %q", restoredPath, "/tmp/original.mp4")
	}

	var start, end float64
	var faceCount, variantID int
	if err := s.pool.QueryRow(ctx, `
		SELECT start_time, end_time, face_count, variant_id
		FROM face_intervals
		WHERE video_id = $1
	`, videoID).Scan(&start, &end, &faceCount, &variantID); err != nil {
		t.Fatalf("failed to load restored interval: %v", err)
	}
	if start != 1.0 || end != 2.5 || faceCount != 3 || variantID != originalVariantID {
		t.Fatalf("restored interval = (%v, %v, %d, %d), want (%v, %v, %d, %d)", start, end, faceCount, variantID, 1.0, 2.5, 3, originalVariantID)
	}

	var variantCount int
	if err := s.pool.QueryRow(ctx, "SELECT COUNT(*) FROM variants").Scan(&variantCount); err != nil {
		t.Fatalf("failed to count variants after rollback: %v", err)
	}
	if variantCount != 1 {
		t.Fatalf("expected rollback to remove newly created variant, got %d variants", variantCount)
	}
}

func TestRevertCommitRejectsOlderVideoCommitWhenNewerOneIsActive(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	ctx := context.Background()

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
		if isDockerUnavailable(err) {
			t.Skipf("Docker not available, skipping integration test: %v", err)
		}
		t.Fatalf("Failed to start postgres container: %v", err)
	}
	defer func() {
		if err := pgContainer.Terminate(ctx); err != nil {
			t.Fatalf("Failed to terminate container: %v", err)
		}
	}()

	connStr, err := pgContainer.ConnectionString(ctx, "sslmode=disable")
	if err != nil {
		t.Fatalf("Failed to get connection string: %v", err)
	}

	s, err := New(ctx, connStr)
	if err != nil {
		t.Fatalf("Failed to connect to store: %v", err)
	}
	defer s.Close(ctx)

	baseVec := make([]float64, 512)
	baseVec[0] = 1.0
	baseVariantID, _, err := s.CreateIdentity(ctx, baseVec, 2)
	if err != nil {
		t.Fatalf("CreateIdentity failed: %v", err)
	}

	const videoID = "vid_shared"
	if err := s.EnsureVideoMetadata(ctx, videoID, "/tmp/original.mp4"); err != nil {
		t.Fatalf("EnsureVideoMetadata failed: %v", err)
	}
	if err := s.InsertInterval(ctx, videoID, 0.0, 1.0, 2, baseVariantID); err != nil {
		t.Fatalf("InsertInterval failed: %v", err)
	}

	vecA := make([]float64, 512)
	vecA[1] = 1.0
	if err := s.ApplyCommitBatch(ctx, "commit_a", []CommitAction{{
		TrackID: "Track_A",
		Action:  "new_identity",
		Vector:  vecA,
		Count:   3,
	}}, videoID, "/tmp/a.mp4", []CommitInterval{{
		TrackID:   "Track_A",
		Start:     5.0,
		End:       6.0,
		FaceCount: 3,
	}}); err != nil {
		t.Fatalf("ApplyCommitBatch(commit_a) failed: %v", err)
	}

	time.Sleep(10 * time.Millisecond)

	vecB := make([]float64, 512)
	vecB[2] = 1.0
	if err := s.ApplyCommitBatch(ctx, "commit_b", []CommitAction{{
		TrackID: "Track_B",
		Action:  "new_identity",
		Vector:  vecB,
		Count:   4,
	}}, videoID, "/tmp/b.mp4", []CommitInterval{{
		TrackID:   "Track_B",
		Start:     8.0,
		End:       9.0,
		FaceCount: 4,
	}}); err != nil {
		t.Fatalf("ApplyCommitBatch(commit_b) failed: %v", err)
	}

	err = s.RevertCommit(ctx, "commit_a")
	if err == nil {
		t.Fatal("expected rollback of older commit to be rejected")
	}
	if !strings.Contains(err.Error(), "newer active commit") {
		t.Fatalf("unexpected rollback error: %v", err)
	}
}

type noopLogger struct{}

func (n noopLogger) Printf(format string, v ...interface{}) {}

func isDockerUnavailable(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "cannot connect to the docker daemon") ||
		strings.Contains(msg, "docker.sock") ||
		strings.Contains(msg, "connection refused") ||
		strings.Contains(msg, "permission denied")
}
