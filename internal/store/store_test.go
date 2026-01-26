package store

import (
	"context"
	"fmt"
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
		t.Fatalf("Docker not available, cannot run integration test: %v", err)
	}

	// Start Postgres Container with pgvector
	// We use the official pgvector image to ensure the extension is available.
	pgContainer, err := postgres.RunContainer(ctx,
		testcontainers.WithImage("pgvector/pgvector:pg16"),
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
	idA, err := s.CreateIdentity(ctx, vecA, 1)
	if err != nil {
		t.Fatalf("CreateIdentity failed: %v", err)
	}
	if idA <= 0 {
		t.Errorf("Expected positive ID, got %d", idA)
	}

	// Find Closest Identity (Exact Match)
	matchID, _, err := s.FindClosestIdentity(ctx, vecA, 0.1)
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
	noMatchID, _, err := s.FindClosestIdentity(ctx, vecB, 0.1)
	if err != nil {
		t.Fatalf("FindClosestIdentity error: %v", err)
	}
	if noMatchID != -1 {
		t.Errorf("Expected no match (-1), got %d", noMatchID)
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
	vectors, err := s.GetIdentityVectors(ctx, []int{idA})
	if err != nil {
		t.Fatalf("GetIdentityVectors failed: %v", err)
	}
	updatedVec, ok := vectors[idA]
	if !ok {
		t.Fatalf("Updated vector for ID %d not found", idA)
	}

	// Check the math
	// Expected average is [0.5, 0.5, 0.0, ...]
	if len(updatedVec) != 512 {
		t.Fatalf("Expected vector of length 512, got %d", len(updatedVec))
	}
	epsilon := 1e-6
	if updatedVec[0] < 0.5-epsilon || updatedVec[0] > 0.5+epsilon {
		t.Errorf("Expected updatedVec[0] to be ~0.5, got %f", updatedVec[0])
	}
	if updatedVec[1] < 0.5-epsilon || updatedVec[1] > 0.5+epsilon {
		t.Errorf("Expected updatedVec[1] to be ~0.5, got %f", updatedVec[1])
	}

	err = s.EnsureVideoMetadata(ctx, "vid_123", "/tmp/video.mp4")
	if err != nil {
		t.Fatalf("EnsureVideoMetadata failed: %v", err)
	}
	err = s.InsertInterval(ctx, "vid_123", 0.0, 5.0, 10, idA)
	if err != nil {
		t.Fatalf("InsertInterval failed: %v", err)
	}

	identities, err := s.ListIdentities(ctx)
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

type noopLogger struct{}

func (n noopLogger) Printf(format string, v ...interface{}) {}
