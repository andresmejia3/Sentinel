package store

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
)

// Store manages the PostgreSQL connection and pgvector operations.
type Store struct {
	conn *pgx.Conn
}

// New establishes a connection to the database and ensures the schema is initialized.
func New(ctx context.Context, connString string) (*Store, error) {
	conn, err := pgx.Connect(ctx, connString)
	if err != nil {
		return nil, err
	}

	// Initialize schema (Auto-Migration)
	if err := initSchema(ctx, conn); err != nil {
		conn.Close(ctx)
		return nil, fmt.Errorf("failed to initialize database schema: %w", err)
	}

	return &Store{conn: conn}, nil
}

// initSchema creates the necessary tables and vector extension if they don't exist (Auto-Migration).
func initSchema(ctx context.Context, conn *pgx.Conn) error {
	query := `
		CREATE EXTENSION IF NOT EXISTS vector;
		CREATE TABLE IF NOT EXISTS video_metadata (
			id TEXT PRIMARY KEY,
			path TEXT NOT NULL,
			indexed_at TIMESTAMPTZ DEFAULT NOW()
		);
		CREATE TABLE IF NOT EXISTS face_intervals (
			id BIGSERIAL PRIMARY KEY, 
			video_id TEXT REFERENCES video_metadata(id),
			start_time DOUBLE PRECISION NOT NULL,
			end_time DOUBLE PRECISION NOT NULL,
			face_count INT NOT NULL,
			known_identity_id INT REFERENCES known_identities(id)
		);
		CREATE TABLE IF NOT EXISTS known_identities (
			id SERIAL PRIMARY KEY,
			name TEXT NOT NULL UNIQUE,
			embedding VECTOR(512) NOT NULL,
			face_count INT DEFAULT 1,
			created_at TIMESTAMPTZ DEFAULT NOW()
		);
		CREATE INDEX IF NOT EXISTS face_intervals_video_id_idx ON face_intervals (video_id);
	`
	_, err := conn.Exec(ctx, query)
	return err
}

// Close terminates the database connection.
func (s *Store) Close(ctx context.Context) {
	s.conn.Close(ctx)
}

// EnsureVideoMetadata registers the video in the database. If it exists, it updates the timestamp.
func (s *Store) EnsureVideoMetadata(ctx context.Context, videoID, path string) error {
	// 1. Clean up old data to ensure idempotency (prevent duplicate intervals on re-scan)
	if _, err := s.conn.Exec(ctx, "DELETE FROM face_intervals WHERE video_id = $1", videoID); err != nil {
		return err
	}

	_, err := s.conn.Exec(ctx, `
		INSERT INTO video_metadata (id, path, indexed_at)
		VALUES ($1, $2, NOW())
		ON CONFLICT (id) DO UPDATE SET indexed_at = NOW(), path = EXCLUDED.path
	`, videoID, path)
	return err
}

// InsertInterval saves a merged face interval to the database.
func (s *Store) InsertInterval(ctx context.Context, videoID string, start, end float64, count, knownID int) error {
	_, err := s.conn.Exec(ctx, `
		INSERT INTO face_intervals (video_id, start_time, end_time, face_count, known_identity_id)
		VALUES ($1, $2, $3, $4, $5)
	`, videoID, start, end, count, knownID)
	return err
}

// vecToString formats a float slice into a PostgreSQL vector string format "[1.0,2.0,...]"
func vecToString(vec []float64) string {
	var b strings.Builder
	b.WriteByte('[')
	for i, v := range vec {
		if i > 0 {
			b.WriteByte(',')
		}
		fmt.Fprintf(&b, "%f", v)
	}
	b.WriteByte(']')
	return b.String()
}

// FindClosestIdentity searches for the nearest neighbor in the database using cosine distance.
// Returns -1 if no match is found within the threshold.
func (s *Store) FindClosestIdentity(ctx context.Context, vec []float64, threshold float64) (int, error) {
	vecStr := vecToString(vec)
	// <=> is the cosine distance operator in pgvector
	// We order by distance and limit to 1 to find the nearest neighbor
	query := `SELECT id FROM known_identities WHERE embedding <=> $1::vector < $2 ORDER BY embedding <=> $1::vector ASC LIMIT 1`

	var id int
	err := s.conn.QueryRow(ctx, query, vecStr, threshold).Scan(&id)
	if err == pgx.ErrNoRows {
		return -1, nil // No match found
	}
	if err != nil {
		return 0, err
	}

	return id, nil
}

// CreateIdentity inserts a new unknown identity and returns its ID.
func (s *Store) CreateIdentity(ctx context.Context, vec []float64) (int, error) {
	vecStr := vecToString(vec)
	var id int
	// We use a temporary unique name to avoid collisions before we know the ID
	tempName := fmt.Sprintf("pending-%d", time.Now().UnixNano())

	tx, err := s.conn.Begin(ctx)
	if err != nil {
		return 0, err
	}
	defer tx.Rollback(ctx)

	// 1. Insert with placeholder
	err = tx.QueryRow(ctx, "INSERT INTO known_identities (name, embedding, face_count) VALUES ($1, $2::vector, 1) RETURNING id", tempName, vecStr).Scan(&id)
	if err != nil {
		return 0, err
	}

	// 2. Update name to "Identity <ID>"
	finalName := fmt.Sprintf("Identity %d", id)
	_, err = tx.Exec(ctx, "UPDATE known_identities SET name = $1 WHERE id = $2", finalName, id)
	if err != nil {
		return 0, err
	}

	return id, tx.Commit(ctx)
}

// UpdateIdentity performs a weighted average update on an existing identity's embedding.
func (s *Store) UpdateIdentity(ctx context.Context, id int, newVec []float64, newCount int) error {
	// 1. Fetch current state
	var oldVecStr string
	var oldCount int
	// FOR UPDATE locks the row to prevent race conditions if multiple workers update the same ID
	err := s.conn.QueryRow(ctx, "SELECT embedding::text, face_count FROM known_identities WHERE id = $1 FOR UPDATE", id).Scan(&oldVecStr, &oldCount)
	if err != nil {
		return err
	}

	// 2. Weighted Math
	oldVecStr = strings.Trim(oldVecStr, "[]")
	parts := strings.Split(oldVecStr, ",")
	finalVec := make([]float64, 512)
	totalCount := float64(oldCount + newCount)

	for i, p := range parts {
		if i < 512 {
			oldVal, _ := strconv.ParseFloat(p, 64)
			finalVec[i] = (oldVal*float64(oldCount) + newVec[i]*float64(newCount)) / totalCount
		}
	}

	finalVecStr := vecToString(finalVec)
	_, err = s.conn.Exec(ctx, "UPDATE known_identities SET embedding = $1::vector, face_count = $2 WHERE id = $3", finalVecStr, int(totalCount), id)
	return err
}

// RenameIdentity updates the name of a known identity.
func (s *Store) RenameIdentity(ctx context.Context, id int, newName string) error {
	_, err := s.conn.Exec(ctx, "UPDATE known_identities SET name = $1 WHERE id = $2", newName, id)
	return err
}

// Reset drops all application tables to clear the database state.
// This is useful for development to force a schema refresh without migrations.
func (s *Store) Reset(ctx context.Context) error {
	_, err := s.conn.Exec(ctx, `
		DROP TABLE IF EXISTS face_intervals CASCADE;
		DROP TABLE IF EXISTS known_identities CASCADE;
		DROP TABLE IF EXISTS video_metadata CASCADE;
	`)
	return err
}
