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
			if pool.Ping(ctx) == nil {
				break
			}
			pool.Close() // Fix: Close the pool if Ping fails to prevent resource leak
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

		CREATE TABLE IF NOT EXISTS identities (
			id SERIAL PRIMARY KEY,
			name TEXT UNIQUE,
			created_at TIMESTAMPTZ DEFAULT NOW()
		);

		CREATE TABLE IF NOT EXISTS variants (
			id SERIAL PRIMARY KEY,
			identity_id INT REFERENCES identities(id) ON DELETE CASCADE,
			name TEXT, -- Variant Name (e.g. "Default", "Glasses")
			embedding VECTOR(512) NOT NULL,
			face_count INT DEFAULT 1,
			created_at TIMESTAMPTZ DEFAULT NOW(),
			UNIQUE(identity_id, name)
		);

		CREATE TABLE IF NOT EXISTS face_intervals (
			id BIGSERIAL PRIMARY KEY, 
			video_id TEXT REFERENCES video_metadata(id),
			start_time DOUBLE PRECISION NOT NULL,
			end_time DOUBLE PRECISION NOT NULL,
			face_count INT NOT NULL,
			variant_id INT REFERENCES variants(id)
		);

		CREATE INDEX IF NOT EXISTS face_intervals_video_id_idx ON face_intervals (video_id);
		CREATE INDEX IF NOT EXISTS face_intervals_variant_id_idx ON face_intervals (variant_id);
		CREATE INDEX IF NOT EXISTS variants_embedding_idx ON variants USING hnsw (embedding vector_cosine_ops);
	`
	_, err := pool.Exec(ctx, query)
	return err
}

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
		INSERT INTO face_intervals (video_id, start_time, end_time, face_count, variant_id)
		VALUES ($1, $2, $3, $4, $5)
	`, videoID, start, end, count, knownID)
	return err
}

// IntervalData holds the data for a single face interval to be committed.
type IntervalData struct {
	Start     float64
	End       float64
	FaceCount int
	VariantID int
}

// CommitScan transactionally replaces all intervals for a given video ID.
// This makes the scan operation atomic: it either completes fully or leaves the old data untouched.
func (s *Store) CommitScan(ctx context.Context, videoID string, intervals []IntervalData) error {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)

	if _, err := tx.Exec(ctx, "DELETE FROM face_intervals WHERE video_id = $1", videoID); err != nil {
		return fmt.Errorf("failed to delete old intervals: %w", err)
	}

	if len(intervals) > 0 {
		batch := &pgx.Batch{}
		for _, i := range intervals {
			batch.Queue(`INSERT INTO face_intervals (video_id, start_time, end_time, face_count, variant_id) VALUES ($1, $2, $3, $4, $5)`, videoID, i.Start, i.End, i.FaceCount, i.VariantID)
		}

		br := tx.SendBatch(ctx, batch)

		for i := 0; i < len(intervals); i++ {
			if _, err := br.Exec(); err != nil {
				br.Close() // Ensure closed on error path
				return fmt.Errorf("batch insert failed on interval %d: %w", i, err)
			}
		}
		if err := br.Close(); err != nil {
			return fmt.Errorf("failed to close batch results: %w", err)
		}
	}
	return tx.Commit(ctx)
}

// FindClosestIdentity searches for the nearest neighbor in the database using cosine distance.
// Returns -1 if no match is found within the threshold.
func (s *Store) FindClosestIdentity(ctx context.Context, vec []float64, threshold float64) (variantID int, masterID int, masterName string, variantName string, err error) {
	// Optimization: Use binary protocol (pass []float32) to avoid string parsing overhead.
	vec32 := make([]float32, len(vec))
	for i, v := range vec {
		vec32[i] = float32(v)
	}

	// <=> is the cosine distance operator in pgvector
	// We order by distance and limit to 1 to find the nearest neighbor
	// We join with identities to return the Master Name
	query := `
		SELECT v.id, i.id, COALESCE(i.name, 'Identity ' || i.id), COALESCE(v.name, 'Default')
		FROM variants v
		LEFT JOIN identities i ON v.identity_id = i.id
		WHERE v.embedding <=> $1::real[]::vector < $2 
		ORDER BY v.embedding <=> $1::real[]::vector ASC LIMIT 1`

	err = s.pool.QueryRow(ctx, query, vec32, threshold).Scan(&variantID, &masterID, &masterName, &variantName)
	if err == pgx.ErrNoRows {
		return -1, 0, "", "", nil // No match found
	}
	if err != nil {
		return 0, 0, "", "", err
	}

	return variantID, masterID, masterName, variantName, nil
}

// VariantData holds the embedding and linkage info for a variant.
type VariantData struct {
	VariantID int
	MasterID  int
	Vec       []float64
}

// GetVariantsForIdentities retrieves all variant embeddings for a list of Master Identity IDs.
func (s *Store) GetVariantsForIdentities(ctx context.Context, masterIDs []int) ([]VariantData, error) {
	if len(masterIDs) == 0 {
		return nil, nil
	}

	// Use ANY($1) to match any ID in the list
	query := `SELECT id, identity_id, embedding::real[] FROM variants WHERE identity_id = ANY($1)`

	rows, err := s.pool.Query(ctx, query, masterIDs)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []VariantData
	for rows.Next() {
		var v VariantData
		var vec32 []float32
		if err := rows.Scan(&v.VariantID, &v.MasterID, &vec32); err != nil {
			return nil, err
		}
		v.Vec = make([]float64, len(vec32))
		for i, val := range vec32 {
			v.Vec[i] = float64(val)
		}
		results = append(results, v)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return results, nil
}

// CreateIdentity inserts a new unknown identity and returns its ID.
func (s *Store) CreateIdentity(ctx context.Context, vec []float64, count int) (variantID int, masterID int, err error) {
	// Optimization: Use binary protocol (pass []float32) to avoid string parsing overhead.
	vec32 := make([]float32, len(vec))
	for i, v := range vec {
		vec32[i] = float32(v)
	}

	tx, err := s.pool.Begin(ctx)
	if err != nil { // Return 0 for both IDs on error
		return 0, 0, err
	}
	defer tx.Rollback(ctx)

	// 1. Create a new Master Identity (Name is NULL initially, will be "Identity <ID>" in UI)
	if err := tx.QueryRow(ctx, "INSERT INTO identities DEFAULT VALUES RETURNING id").Scan(&masterID); err != nil {
		return 0, 0, err // Return 0 for both IDs on error
	}

	// 2. Create the Default Variant
	err = tx.QueryRow(ctx, "INSERT INTO variants (embedding, face_count, identity_id, name) VALUES ($1::real[]::vector, $2, $3, 'Default') RETURNING id", vec32, count, masterID).Scan(&variantID)
	if err != nil {
		return 0, 0, err // Return 0 for both IDs on error
	}

	return variantID, masterID, tx.Commit(ctx)
}

// MasterIdentityExists checks if a master identity exists in the database.
func (s *Store) MasterIdentityExists(ctx context.Context, masterID int) (bool, error) {
	var exists bool
	err := s.pool.QueryRow(ctx, "SELECT EXISTS(SELECT 1 FROM identities WHERE id = $1)", masterID).Scan(&exists)
	return exists, err
}

// GetMasterIDForVariant retrieves the Master Identity ID for a given Variant ID.
func (s *Store) GetMasterIDForVariant(ctx context.Context, variantID int) (int, error) {
	var masterID int
	err := s.pool.QueryRow(ctx, "SELECT identity_id FROM variants WHERE id = $1", variantID).Scan(&masterID)
	if err == pgx.ErrNoRows {
		return 0, fmt.Errorf("variant %d not found", variantID)
	}
	return masterID, err
}

// SetVariantLabel links an existing variant to a Master Identity (creating it if needed) and names the variant.
func (s *Store) SetVariantLabel(ctx context.Context, variantID int, masterName string, variantName string) error {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)

	// Upsert Master Identity
	var masterID int
	err = tx.QueryRow(ctx, "INSERT INTO identities (name) VALUES ($1) ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name RETURNING id", masterName).Scan(&masterID)
	if err != nil {
		return fmt.Errorf("failed to get/create master identity: %w", err)
	}

	// Update Variant
	_, err = tx.Exec(ctx, "UPDATE variants SET identity_id = $1, name = $2 WHERE id = $3", masterID, variantName, variantID)
	if err != nil {
		return fmt.Errorf("failed to update variant: %w", err)
	}
	return tx.Commit(ctx)
}

// UpdateIdentity performs a weighted average update on an existing identity's embedding.
func (s *Store) UpdateIdentity(ctx context.Context, id int, newVec []float64, newCount int) error {
	// Transaction is required to hold the FOR UPDATE lock across the read and write
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)

	// Optimization: Cast to real[] to let pgx scan directly into []float32 (Binary Protocol)
	// This avoids the expensive string parsing of "[0.1, 0.2...]"
	var oldVec []float32
	var oldCount int
	err = tx.QueryRow(ctx, "SELECT embedding::real[], face_count FROM variants WHERE id = $1 FOR UPDATE", id).Scan(&oldVec, &oldCount)
	if err != nil {
		return err
	}

	totalCount := float64(oldCount + newCount)
	finalVec := make([]float32, len(oldVec))

	for i := range oldVec {
		val := (float64(oldVec[i])*float64(oldCount) + newVec[i]*float64(newCount)) / totalCount
		finalVec[i] = float32(val)
	}

	// We pass the []float32 slice directly. pgx sends it as a float array, and Postgres casts it to vector.
	_, err = tx.Exec(ctx, "UPDATE variants SET embedding = $1::real[]::vector, face_count = $2 WHERE id = $3", finalVec, int(totalCount), id)
	if err != nil {
		return err
	}
	return tx.Commit(ctx)
}

// RenameIdentity renames a Master Identity directly by its ID.
func (s *Store) RenameIdentity(ctx context.Context, masterID int, newName string) error {
	_, err := s.pool.Exec(ctx, "UPDATE identities SET name = $1 WHERE id = $2", newName, masterID)
	return err
}

// DeleteIdentity removes an identity from the database.
// Used for cleaning up "ghost" identities that were created but filtered out as blips.
func (s *Store) DeleteIdentity(ctx context.Context, id int) error {
	_, err := s.pool.Exec(ctx, "DELETE FROM variants WHERE id = $1", id)
	return err
}

// Reset drops all application tables to clear the database state.
// This is useful for development to force a schema refresh without migrations.
func (s *Store) Reset(ctx context.Context) error {
	_, err := s.pool.Exec(ctx, `
		DROP TABLE IF EXISTS face_intervals CASCADE;
		DROP TABLE IF EXISTS variants CASCADE;
		DROP TABLE IF EXISTS identities CASCADE;
		DROP TABLE IF EXISTS video_metadata CASCADE;
	`)
	return err
}

// IdentityMetadata represents the display info for an identity
type IdentityMetadata struct {
	ID           int
	Name         string
	Count        int // Total face count across all variants
	VariantCount int // Number of variants for this master
	CreatedAt    time.Time
}

type VariantMetadata struct {
	ID        int
	Name      string
	Count     int // Face count for this specific variant
	CreatedAt time.Time
}

func (s *Store) ListIdentities(ctx context.Context) ([]IdentityMetadata, error) {
	// List Master Identities
	// We LEFT JOIN so that master identities with no variants yet still appear.
	// We SUM the face_count from all variants belonging to a master identity.
	query := `
		SELECT
			i.id,
			COALESCE(i.name, 'Identity ' || i.id),
			COALESCE(SUM(v.face_count), 0)::INT,
			COUNT(v.id)::INT,
			i.created_at
		FROM identities i
		LEFT JOIN variants v ON i.id = v.identity_id
		GROUP BY i.id, i.name, i.created_at ORDER BY i.id ASC`
	rows, err := s.pool.Query(ctx, query)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []IdentityMetadata
	for rows.Next() {
		var i IdentityMetadata
		if err := rows.Scan(&i.ID, &i.Name, &i.Count, &i.VariantCount, &i.CreatedAt); err != nil {
			return nil, err
		}
		results = append(results, i)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return results, nil
}

func (s *Store) ListVariantsForIdentity(ctx context.Context, masterID int) ([]VariantMetadata, error) {
	query := `
		SELECT id, COALESCE(name, 'Default'), face_count, created_at
		FROM variants
		WHERE identity_id = $1
		ORDER BY id ASC`
	rows, err := s.pool.Query(ctx, query, masterID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []VariantMetadata
	for rows.Next() {
		var v VariantMetadata
		if err := rows.Scan(&v.ID, &v.Name, &v.Count, &v.CreatedAt); err != nil {
			return nil, err
		}
		results = append(results, v)
	}
	if err := rows.Err(); err != nil {
		return nil, err
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

func (s *Store) GetIdentityIntervals(ctx context.Context, identityID int) ([]IntervalResult, error) {
	query := `
		SELECT f.video_id, v.path, f.start_time, f.end_time
		FROM face_intervals f
		JOIN video_metadata v ON f.video_id = v.id
		WHERE f.variant_id = $1
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
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return results, nil
}
