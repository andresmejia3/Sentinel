package store

import (
	"context"
	"fmt"
	"strings"

	"github.com/andresmejia3/sentinel/internal/types"
	"github.com/jackc/pgx/v5"
)

// Store manages the PostgreSQL connection and pgvector operations.
type Store struct {
	conn *pgx.Conn
}

// New establishes a connection to the database.
func New(ctx context.Context, connString string) (*Store, error) {
	conn, err := pgx.Connect(ctx, connString)
	if err != nil {
		return nil, err
	}
	return &Store{conn: conn}, nil
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

// InsertDetections saves a batch of face embeddings and their locations to the database.
func (s *Store) InsertDetections(ctx context.Context, videoID string, frameIdx int, faces []types.FaceResult) error {
	if len(faces) == 0 {
		return nil
	}

	// Prepare batch insert (or simple loop for now)
	for _, face := range faces {
		vecStr := vecToString(face.Vec)
		_, err := s.conn.Exec(ctx, `
			INSERT INTO face_detections (video_id, frame_index, location, embedding)
			VALUES ($1, $2, $3, $4::vector)
		`, videoID, frameIdx, face.Loc, vecStr)
		if err != nil {
			return err
		}
	}
	return nil
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
