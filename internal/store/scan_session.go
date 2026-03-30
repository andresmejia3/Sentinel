package store

import (
	"context"
	"fmt"

	"github.com/jackc/pgx/v5"
)

// ScanSession keeps all scan-side DB mutations inside one transaction until finalize.
type ScanSession struct {
	tx     pgx.Tx
	closed bool
}

// BeginScanSession starts a transaction-backed session for a single scan.
func (s *Store) BeginScanSession(ctx context.Context) (*ScanSession, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return nil, err
	}
	return &ScanSession{tx: tx}, nil
}

// Rollback aborts the scan transaction if it has not been finalized yet.
func (s *ScanSession) Rollback(ctx context.Context) error {
	if s == nil || s.closed {
		return nil
	}
	s.closed = true
	return s.tx.Rollback(ctx)
}

// FindClosestIdentity searches for the nearest neighbor inside the active scan transaction.
func (s *ScanSession) FindClosestIdentity(ctx context.Context, vec []float64, threshold float64) (variantID int, identityID int, identityName string, variantName string, err error) {
	vec32 := toFloat32Slice(vec)

	query := `
		SELECT v.id, i.id, COALESCE(i.name, 'Identity ' || i.id), COALESCE(v.name, 'Default')
		FROM variants v
		LEFT JOIN identities i ON v.identity_id = i.id
		WHERE v.embedding <=> $1::real[]::vector < $2
		ORDER BY v.embedding <=> $1::real[]::vector ASC LIMIT 1`

	err = s.tx.QueryRow(ctx, query, vec32, threshold).Scan(&variantID, &identityID, &identityName, &variantName)
	if err == pgx.ErrNoRows {
		return -1, 0, "", "", nil
	}
	if err != nil {
		return 0, 0, "", "", err
	}

	return variantID, identityID, identityName, variantName, nil
}

// FindTopIdentities returns the top K nearest neighbors inside the active scan transaction.
func (s *ScanSession) FindTopIdentities(ctx context.Context, vec []float64, limit int) ([]IdentityMatch, error) {
	vec32 := toFloat32Slice(vec)

	query := `
		SELECT v.id, i.id, COALESCE(i.name, 'Identity ' || i.id), COALESCE(v.name, 'Default'), v.embedding <=> $1::real[]::vector
		FROM variants v
		LEFT JOIN identities i ON v.identity_id = i.id
		ORDER BY v.embedding <=> $1::real[]::vector ASC LIMIT $2`

	rows, err := s.tx.Query(ctx, query, vec32, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var matches []IdentityMatch
	for rows.Next() {
		var m IdentityMatch
		if err := rows.Scan(&m.VariantID, &m.IdentityID, &m.IdentityName, &m.VariantName, &m.Distance); err != nil {
			return nil, err
		}
		matches = append(matches, m)
	}
	return matches, rows.Err()
}

// CreateIdentity inserts a new identity and its default variant inside the active scan transaction.
func (s *ScanSession) CreateIdentity(ctx context.Context, vec []float64, count int) (variantID int, identityID int, err error) {
	vec32 := toFloat32Slice(vec)

	if err := s.tx.QueryRow(ctx, "INSERT INTO identities DEFAULT VALUES RETURNING id").Scan(&identityID); err != nil {
		return 0, 0, err
	}
	err = s.tx.QueryRow(ctx, "INSERT INTO variants (embedding, face_count, identity_id, name) VALUES ($1::real[]::vector, $2, $3, 'Default') RETURNING id", vec32, count, identityID).Scan(&variantID)
	if err != nil {
		return 0, 0, err
	}
	return variantID, identityID, nil
}

// UpdateIdentity updates a variant embedding inside the active scan transaction.
func (s *ScanSession) UpdateIdentity(ctx context.Context, id int, newVec []float64, newCount int) error {
	var oldVec []float32
	var oldCount int
	err := s.tx.QueryRow(ctx, "SELECT embedding::real[], face_count FROM variants WHERE id = $1 FOR UPDATE", id).Scan(&oldVec, &oldCount)
	if err != nil {
		return err
	}
	if len(newVec) != len(oldVec) {
		return fmt.Errorf("vector dimension mismatch: expected %d, got %d", len(oldVec), len(newVec))
	}

	totalCount := float64(oldCount + newCount)
	finalVec := make([]float32, len(oldVec))
	for i := range oldVec {
		val := (float64(oldVec[i])*float64(oldCount) + newVec[i]*float64(newCount)) / totalCount
		finalVec[i] = float32(val)
	}

	_, err = s.tx.Exec(ctx, "UPDATE variants SET embedding = $1::real[]::vector, face_count = $2 WHERE id = $3", finalVec, int(totalCount), id)
	return err
}

// FinalizeScan atomically replaces the intervals for the video and commits the session.
func (s *ScanSession) FinalizeScan(ctx context.Context, videoID string, intervals []IntervalData) error {
	if s == nil || s.closed {
		return fmt.Errorf("scan session is already closed")
	}

	if _, err := s.tx.Exec(ctx, "DELETE FROM face_intervals WHERE video_id = $1", videoID); err != nil {
		return fmt.Errorf("failed to delete old intervals: %w", err)
	}

	if len(intervals) > 0 {
		batch := &pgx.Batch{}
		for _, i := range intervals {
			batch.Queue(`INSERT INTO face_intervals (video_id, start_time, end_time, face_count, variant_id) VALUES ($1, $2, $3, $4, $5)`, videoID, i.Start, i.End, i.FaceCount, i.VariantID)
		}

		br := s.tx.SendBatch(ctx, batch)
		for i := 0; i < len(intervals); i++ {
			if _, err := br.Exec(); err != nil {
				br.Close()
				return fmt.Errorf("batch insert failed on interval %d: %w", i, err)
			}
		}
		if err := br.Close(); err != nil {
			return fmt.Errorf("failed to close batch results: %w", err)
		}
	}

	s.closed = true
	return s.tx.Commit(ctx)
}

func toFloat32Slice(vec []float64) []float32 {
	vec32 := make([]float32, len(vec))
	for i, v := range vec {
		vec32[i] = float32(v)
	}
	return vec32
}
