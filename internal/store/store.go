package store

import (
	"context"
	"fmt"
	"strings"

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
			embedding VECTOR(128) NOT NULL
		);
		CREATE INDEX IF NOT EXISTS face_intervals_embedding_idx ON face_intervals USING hnsw (embedding vector_cosine_ops);
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
	_, err := s.conn.Exec(ctx, `
		INSERT INTO video_metadata (id, path, indexed_at)
		VALUES ($1, $2, NOW())
		ON CONFLICT (id) DO UPDATE SET indexed_at = NOW()
	`, videoID, path)
	return err
}

// InsertInterval saves a merged face interval to the database.
func (s *Store) InsertInterval(ctx context.Context, videoID string, start, end float64, count int, vec []float64) error {
	vecStr := vecToString(vec)
	_, err := s.conn.Exec(ctx, `
		INSERT INTO face_intervals (video_id, start_time, end_time, face_count, embedding)
		VALUES ($1, $2, $3, $4, $5::vector)
	`, videoID, start, end, count, vecStr)
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
