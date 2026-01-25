package store

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Store manages the PostgreSQL connection and pgvector operations.
type Store struct {
	pool *pgxpool.Pool
}

// New establishes a connection to the database and ensures the schema is initialized.
func New(ctx context.Context, connString string) (*Store, error) {
	var pool *pgxpool.Pool
	var err error

	// Retry loop: Wait for DB to become ready (e.g. if in recovery mode)
	for i := 0; i < 15; i++ {
		pool, err = pgxpool.New(ctx, connString)
		if err == nil {
			break
		}
		fmt.Fprintf(os.Stderr, "⏳ Database not ready (Attempt %d/15). Retrying in 2s...\n", i+1)
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(2 * time.Second):
			continue
		}
	}
	if err != nil {
		return nil, fmt.Errorf("database connection failed after retries: %w", err)
	}

	// Initialize schema (Auto-Migration)
	if err := initSchema(ctx, pool); err != nil {
		pool.Close()
		return nil, fmt.Errorf("failed to initialize database schema: %w", err)
	}

	return &Store{pool: pool}, nil
}

// initSchema creates the necessary tables and vector extension if they don't exist (Auto-Migration).
func initSchema(ctx context.Context, pool *pgxpool.Pool) error {
	query := `
		CREATE EXTENSION IF NOT EXISTS vector;
		CREATE TABLE IF NOT EXISTS video_metadata (
			id TEXT PRIMARY KEY,
			path TEXT NOT NULL,
			indexed_at TIMESTAMPTZ DEFAULT NOW()
		);
		CREATE TABLE IF NOT EXISTS known_identities (
			id SERIAL PRIMARY KEY,
			name TEXT UNIQUE,
			embedding VECTOR(512) NOT NULL,
			face_count INT DEFAULT 1,
			created_at TIMESTAMPTZ DEFAULT NOW()
		);
		CREATE TABLE IF NOT EXISTS face_intervals (
			id BIGSERIAL PRIMARY KEY, 
			video_id TEXT REFERENCES video_metadata(id),
			start_time DOUBLE PRECISION NOT NULL,
			end_time DOUBLE PRECISION NOT NULL,
			face_count INT NOT NULL,
			known_identity_id INT REFERENCES known_identities(id)
		);
		CREATE INDEX IF NOT EXISTS face_intervals_video_id_idx ON face_intervals (video_id);
		CREATE INDEX IF NOT EXISTS face_intervals_known_identity_id_idx ON face_intervals (known_identity_id);
		CREATE INDEX IF NOT EXISTS known_identities_embedding_idx ON known_identities USING hnsw (embedding vector_cosine_ops);
	`
	_, err := pool.Exec(ctx, query)
	return err
}

// Close terminates the database connection.
func (s *Store) Close(ctx context.Context) {
	s.pool.Close()
}

// EnsureVideoMetadata registers the video in the database. If it exists, it updates the timestamp.
func (s *Store) EnsureVideoMetadata(ctx context.Context, videoID, path string) error {
	_, err := s.pool.Exec(ctx, `
		INSERT INTO video_metadata (id, path, indexed_at)
		VALUES ($1, $2, NOW())
		ON CONFLICT (id) DO UPDATE SET indexed_at = NOW(), path = EXCLUDED.path
	`, videoID, path)
	return err
}

// InsertInterval saves a merged face interval to the database.
func (s *Store) InsertInterval(ctx context.Context, videoID string, start, end float64, count, knownID int) error {
	_, err := s.pool.Exec(ctx, `
		INSERT INTO face_intervals (video_id, start_time, end_time, face_count, known_identity_id)
		VALUES ($1, $2, $3, $4, $5)
	`, videoID, start, end, count, knownID)
	return err
}

// IntervalData holds the data for a single face interval to be committed.
type IntervalData struct {
	Start           float64
	End             float64
	FaceCount       int
	KnownIdentityID int
}

// CommitScan transactionally replaces all intervals for a given video ID.
// This makes the scan operation atomic: it either completes fully or leaves the old data untouched.
func (s *Store) CommitScan(ctx context.Context, videoID string, intervals []IntervalData) error {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)

	// 1. Delete old intervals for this video
	if _, err := tx.Exec(ctx, "DELETE FROM face_intervals WHERE video_id = $1", videoID); err != nil {
		return fmt.Errorf("failed to delete old intervals: %w", err)
	}

	// 2. Batch insert new intervals
	if len(intervals) > 0 {
		batch := &pgx.Batch{}
		for _, i := range intervals {
			batch.Queue(`INSERT INTO face_intervals (video_id, start_time, end_time, face_count, known_identity_id) VALUES ($1, $2, $3, $4, $5)`, videoID, i.Start, i.End, i.FaceCount, i.KnownIdentityID)
		}

		br := tx.SendBatch(ctx, batch)
		defer br.Close()

		for i := 0; i < len(intervals); i++ {
			if _, err := br.Exec(); err != nil {
				return fmt.Errorf("batch insert failed on interval %d: %w", i, err)
			}
		}
	}
	return tx.Commit(ctx)
}

// FindClosestIdentity searches for the nearest neighbor in the database using cosine distance.
// Returns -1 if no match is found within the threshold.
func (s *Store) FindClosestIdentity(ctx context.Context, vec []float64, threshold float64) (int, string, error) {
	// Optimization: Use binary protocol (pass []float32) to avoid string parsing overhead.
	vec32 := make([]float32, len(vec))
	for i, v := range vec {
		vec32[i] = float32(v)
	}

	// <=> is the cosine distance operator in pgvector
	// We order by distance and limit to 1 to find the nearest neighbor
	query := `SELECT id, COALESCE(name, '') FROM known_identities WHERE embedding <=> $1::real[]::vector < $2 ORDER BY embedding <=> $1::real[]::vector ASC LIMIT 1`

	var id int
	var name string
	err := s.pool.QueryRow(ctx, query, vec32, threshold).Scan(&id, &name)
	if err == pgx.ErrNoRows {
		return -1, "", nil // No match found
	}
	if err != nil {
		return 0, "", err
	}

	return id, name, nil
}

// GetIdentityVectors retrieves the embeddings for a specific list of identity IDs.
func (s *Store) GetIdentityVectors(ctx context.Context, ids []int) (map[int][]float64, error) {
	if len(ids) == 0 {
		return map[int][]float64{}, nil
	}

	// Use ANY($1) to match any ID in the list
	query := `SELECT id, embedding::real[] FROM known_identities WHERE id = ANY($1)`

	rows, err := s.pool.Query(ctx, query, ids)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	results := make(map[int][]float64)
	for rows.Next() {
		var id int
		var vec32 []float32
		if err := rows.Scan(&id, &vec32); err != nil {
			return nil, err
		}
		// Convert float32 (db) to float64 (app)
		vec64 := make([]float64, len(vec32))
		for i, v := range vec32 {
			vec64[i] = float64(v)
		}
		results[id] = vec64
	}
	return results, nil
}

// CreateIdentity inserts a new unknown identity and returns its ID.
func (s *Store) CreateIdentity(ctx context.Context, vec []float64, count int) (int, error) {
	// Optimization: Use binary protocol (pass []float32) to avoid string parsing overhead.
	vec32 := make([]float32, len(vec))
	for i, v := range vec {
		vec32[i] = float32(v)
	}

	var id int
	// Optimization: We insert with a NULL name (Unknown).
	// The application layer handles formatting "Identity <ID>" when displaying.
	err := s.pool.QueryRow(ctx, "INSERT INTO known_identities (embedding, face_count) VALUES ($1::real[]::vector, $2) RETURNING id", vec32, count).Scan(&id)
	return id, err
}

// UpdateIdentity performs a weighted average update on an existing identity's embedding.
func (s *Store) UpdateIdentity(ctx context.Context, id int, newVec []float64, newCount int) error {
	// Transaction is required to hold the FOR UPDATE lock across the read and write
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)

	// 1. Fetch current state
	// Optimization: Cast to real[] to let pgx scan directly into []float32 (Binary Protocol)
	// This avoids the expensive string parsing of "[0.1, 0.2...]"
	var oldVec []float32
	var oldCount int
	err = tx.QueryRow(ctx, "SELECT embedding::real[], face_count FROM known_identities WHERE id = $1 FOR UPDATE", id).Scan(&oldVec, &oldCount)
	if err != nil {
		return err
	}

	// 2. Weighted Math
	totalCount := float64(oldCount + newCount)
	finalVec := make([]float32, len(oldVec))

	for i := range oldVec {
		val := (float64(oldVec[i])*float64(oldCount) + newVec[i]*float64(newCount)) / totalCount
		finalVec[i] = float32(val)
	}

	// 3. Update
	// We pass the []float32 slice directly. pgx sends it as a float array, and Postgres casts it to vector.
	_, err = tx.Exec(ctx, "UPDATE known_identities SET embedding = $1::real[]::vector, face_count = $2 WHERE id = $3", finalVec, int(totalCount), id)
	if err != nil {
		return err
	}
	return tx.Commit(ctx)
}

// RenameIdentity updates the name of a known identity.
func (s *Store) RenameIdentity(ctx context.Context, id int, newName string) error {
	_, err := s.pool.Exec(ctx, "UPDATE known_identities SET name = $1 WHERE id = $2", newName, id)
	return err
}

// DeleteIdentity removes an identity from the database.
// Used for cleaning up "ghost" identities that were created but filtered out as blips.
func (s *Store) DeleteIdentity(ctx context.Context, id int) error {
	_, err := s.pool.Exec(ctx, "DELETE FROM known_identities WHERE id = $1", id)
	return err
}

// Reset drops all application tables to clear the database state.
// This is useful for development to force a schema refresh without migrations.
func (s *Store) Reset(ctx context.Context) error {
	_, err := s.pool.Exec(ctx, `
		DROP TABLE IF EXISTS face_intervals CASCADE;
		DROP TABLE IF EXISTS known_identities CASCADE;
		DROP TABLE IF EXISTS video_metadata CASCADE;
	`)
	return err
}

// IdentityMetadata represents the display info for an identity
type IdentityMetadata struct {
	ID        int
	Name      string
	Count     int
	CreatedAt time.Time
}

// ListIdentities returns all known identities for display.
func (s *Store) ListIdentities(ctx context.Context) ([]IdentityMetadata, error) {
	rows, err := s.pool.Query(ctx, "SELECT id, COALESCE(name, ''), face_count, created_at FROM known_identities ORDER BY id ASC")
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []IdentityMetadata
	for rows.Next() {
		var i IdentityMetadata
		if err := rows.Scan(&i.ID, &i.Name, &i.Count, &i.CreatedAt); err != nil {
			return nil, err
		}
		results = append(results, i)
	}
	return results, nil
}

// IntervalResult holds metadata about a specific appearance of an identity in a video.
type IntervalResult struct {
	VideoID   string
	VideoPath string
	Start     float64
	End       float64
}

// GetIdentityIntervals retrieves all time intervals for a specific identity ID.
func (s *Store) GetIdentityIntervals(ctx context.Context, identityID int) ([]IntervalResult, error) {
	query := `
		SELECT f.video_id, v.path, f.start_time, f.end_time
		FROM face_intervals f
		JOIN video_metadata v ON f.video_id = v.id
		WHERE f.known_identity_id = $1
		ORDER BY v.path, f.start_time
	`
	rows, err := s.pool.Query(ctx, query, identityID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []IntervalResult
	for rows.Next() {
		var r IntervalResult
		if err := rows.Scan(&r.VideoID, &r.VideoPath, &r.Start, &r.End); err != nil {
			return nil, err
		}
		results = append(results, r)
	}
	return results, nil
}
