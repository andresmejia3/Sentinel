package database

import (
	"context"
	"fmt"
	"strings"

	"github.com/andresmejia3/sentinel/internal/types"
	"github.com/jackc/pgx/v5"
)

type Store struct {
	conn *pgx.Conn
}

func New(ctx context.Context, connString string) (*Store, error) {
	conn, err := pgx.Connect(ctx, connString)
	if err != nil {
		return nil, err
	}
	return &Store{conn: conn}, nil
}

func (s *Store) Close(ctx context.Context) {
	s.conn.Close(ctx)
}

func (s *Store) EnsureVideoMetadata(ctx context.Context, videoID, path string) error {
	_, err := s.conn.Exec(ctx, `
		INSERT INTO video_metadata (id, path, indexed_at)
		VALUES ($1, $2, NOW())
		ON CONFLICT (id) DO UPDATE SET indexed_at = NOW()
	`, videoID, path)
	return err
}

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
