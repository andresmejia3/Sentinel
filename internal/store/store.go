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
			variant_id INT REFERENCES variants(id) ON DELETE CASCADE
		);

		CREATE INDEX IF NOT EXISTS face_intervals_video_id_idx ON face_intervals (video_id);
		CREATE INDEX IF NOT EXISTS face_intervals_variant_id_idx ON face_intervals (variant_id);
		CREATE INDEX IF NOT EXISTS variants_embedding_idx ON variants USING hnsw (embedding vector_cosine_ops);

		CREATE TABLE IF NOT EXISTS ledger_entries (
			id BIGSERIAL PRIMARY KEY,
			commit_id TEXT NOT NULL,
			track_id TEXT NOT NULL,
			variant_id INT REFERENCES variants(id) ON DELETE CASCADE,
			added_sum VECTOR(512) NOT NULL,
			added_count INT NOT NULL,
			created_at TIMESTAMPTZ DEFAULT NOW()
		);
		CREATE INDEX IF NOT EXISTS ledger_entries_commit_id_idx ON ledger_entries (commit_id);

		CREATE TABLE IF NOT EXISTS commits (
			id TEXT PRIMARY KEY,
			status TEXT NOT NULL DEFAULT 'processing', -- 'processing', 'active', 'rolling_back', 'rolled_back', 'failed'
			created_at TIMESTAMPTZ DEFAULT NOW()
		);
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
func (s *Store) FindClosestIdentity(ctx context.Context, vec []float64, threshold float64) (variantID int, identityID int, identityName string, variantName string, err error) {
	// Optimization: Use binary protocol (pass []float32) to avoid string parsing overhead.
	vec32 := make([]float32, len(vec))
	for i, v := range vec {
		vec32[i] = float32(v)
	}

	// <=> is the cosine distance operator in pgvector
	// We order by distance and limit to 1 to find the nearest neighbor
	// We join with identities to return the Identity Name
	query := `
		SELECT v.id, i.id, COALESCE(i.name, 'Identity ' || i.id), COALESCE(v.name, 'Default')
		FROM variants v
		LEFT JOIN identities i ON v.identity_id = i.id
		WHERE v.embedding <=> $1::real[]::vector < $2 
		ORDER BY v.embedding <=> $1::real[]::vector ASC LIMIT 1`

	err = s.pool.QueryRow(ctx, query, vec32, threshold).Scan(&variantID, &identityID, &identityName, &variantName)
	if err == pgx.ErrNoRows {
		return -1, 0, "", "", nil // No match found
	}
	if err != nil {
		return 0, 0, "", "", err
	}

	return variantID, identityID, identityName, variantName, nil
}

// IdentityMatch represents a candidate match for k-NN search.
type IdentityMatch struct {
	VariantID    int
	IdentityID   int
	IdentityName string
	VariantName  string
	Distance     float64
}

// FindTopIdentities returns the top K nearest neighbors for ranked k-NN logic.
func (s *Store) FindTopIdentities(ctx context.Context, vec []float64, limit int) ([]IdentityMatch, error) {
	vec32 := make([]float32, len(vec))
	for i, v := range vec {
		vec32[i] = float32(v)
	}

	query := `
		SELECT v.id, i.id, COALESCE(i.name, 'Identity ' || i.id), COALESCE(v.name, 'Default'), v.embedding <=> $1::real[]::vector
		FROM variants v
		LEFT JOIN identities i ON v.identity_id = i.id
		ORDER BY v.embedding <=> $1::real[]::vector ASC LIMIT $2`

	rows, err := s.pool.Query(ctx, query, vec32, limit)
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

// CommitMetadata summarizes a batch transaction.
type CommitMetadata struct {
	CommitID   string
	Status     string
	TotalFaces int
	TrackCount int
	CreatedAt  time.Time
}

// ListCommits retrieves the history of batch operations.
func (s *Store) ListCommits(ctx context.Context, limit int) ([]CommitMetadata, error) {
	var rows pgx.Rows
	var err error

	queryBase := `
		SELECT c.id, c.status, COALESCE(SUM(l.added_count), 0)::INT, COUNT(l.id)::INT, c.created_at
		FROM commits c
		LEFT JOIN ledger_entries l ON c.id = l.commit_id
		GROUP BY c.id, c.status, c.created_at
		ORDER BY c.created_at DESC
	`
	if limit > 0 {
		rows, err = s.pool.Query(ctx, queryBase+" LIMIT $1", limit)
	} else {
		rows, err = s.pool.Query(ctx, queryBase)
	}

	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []CommitMetadata
	for rows.Next() {
		var c CommitMetadata
		if err := rows.Scan(&c.CommitID, &c.Status, &c.TotalFaces, &c.TrackCount, &c.CreatedAt); err != nil {
			return nil, err
		}
		results = append(results, c)
	}
	return results, nil
}

// CreateCommit registers a new batch transaction.
func (s *Store) CreateCommit(ctx context.Context, commitID string) error {
	_, err := s.pool.Exec(ctx, "INSERT INTO commits (id, status) VALUES ($1, 'processing')", commitID)
	return err
}

// GetCommitStatus retrieves the current status of a commit.
func (s *Store) GetCommitStatus(ctx context.Context, commitID string) (string, error) {
	var status string
	err := s.pool.QueryRow(ctx, "SELECT status FROM commits WHERE id = $1", commitID).Scan(&status)
	if err == pgx.ErrNoRows {
		return "", nil // Not found
	}
	return status, err
}

// MarkCommitRolledBack updates the status of a commit to prevent re-rolling.
func (s *Store) MarkCommitRolledBack(ctx context.Context, commitID string) error {
	_, err := s.pool.Exec(ctx, "UPDATE commits SET status = 'rolled_back' WHERE id = $1", commitID)
	return err
}

// MarkCommitStatus updates the status of a commit.
func (s *Store) MarkCommitStatus(ctx context.Context, commitID, status string) error {
	_, err := s.pool.Exec(ctx, "UPDATE commits SET status = $1 WHERE id = $2", status, commitID)
	return err
}

// VariantData holds the embedding and linkage info for a variant.
type VariantData struct {
	VariantID  int
	IdentityID int
	Vec        []float64
}

// GetVariantsForIdentities retrieves all variant embeddings for a list of Identity IDs.
func (s *Store) GetVariantsForIdentities(ctx context.Context, identityIDs []int) ([]VariantData, error) {
	if len(identityIDs) == 0 {
		return nil, nil
	}

	// Use ANY($1) to match any ID in the list
	query := `SELECT id, identity_id, embedding::real[] FROM variants WHERE identity_id = ANY($1)`

	rows, err := s.pool.Query(ctx, query, identityIDs)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []VariantData
	for rows.Next() {
		var v VariantData
		var vec32 []float32
		if err := rows.Scan(&v.VariantID, &v.IdentityID, &vec32); err != nil {
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
func (s *Store) CreateIdentity(ctx context.Context, vec []float64, count int) (variantID int, identityID int, err error) {
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

	// 1. Create a new Identity (Name is NULL initially, will be "Identity <ID>" in UI)
	if err := tx.QueryRow(ctx, "INSERT INTO identities DEFAULT VALUES RETURNING id").Scan(&identityID); err != nil {
		return 0, 0, err // Return 0 for both IDs on error
	}

	// 2. Create the Default Variant
	err = tx.QueryRow(ctx, "INSERT INTO variants (embedding, face_count, identity_id, name) VALUES ($1::real[]::vector, $2, $3, 'Default') RETURNING id", vec32, count, identityID).Scan(&variantID)
	if err != nil {
		return 0, 0, err // Return 0 for both IDs on error
	}

	return variantID, identityID, tx.Commit(ctx)
}

// CreateVariant adds a new variant to an existing identity.
func (s *Store) CreateVariant(ctx context.Context, identityID int, vec []float64, count int, name string) (int, error) {
	vec32 := make([]float32, len(vec))
	for i, v := range vec {
		vec32[i] = float32(v)
	}

	var id int
	err := s.pool.QueryRow(ctx,
		"INSERT INTO variants (identity_id, embedding, face_count, name) VALUES ($1, $2::real[]::vector, $3, $4) RETURNING id",
		identityID, vec32, count, name).Scan(&id)
	return id, err
}

// GetIdentityIDByName looks up an identity ID by its unique name.
func (s *Store) GetIdentityIDByName(ctx context.Context, name string) (int, error) {
	var id int
	err := s.pool.QueryRow(ctx, "SELECT id FROM identities WHERE name = $1", name).Scan(&id)
	if err == pgx.ErrNoRows {
		return 0, nil
	}
	return id, err
}

// GetVariantID looks up a variant ID by identity ID and variant name.
func (s *Store) GetVariantID(ctx context.Context, identityID int, name string) (int, error) {
	var id int
	err := s.pool.QueryRow(ctx, "SELECT id FROM variants WHERE identity_id = $1 AND name = $2", identityID, name).Scan(&id)
	if err == pgx.ErrNoRows {
		return 0, nil
	}
	return id, err
}

// IdentityExists checks if an identity exists in the database.
func (s *Store) IdentityExists(ctx context.Context, identityID int) (bool, error) {
	var exists bool
	err := s.pool.QueryRow(ctx, "SELECT EXISTS(SELECT 1 FROM identities WHERE id = $1)", identityID).Scan(&exists)
	return exists, err
}

// GetIdentityIDForVariant retrieves the Identity ID for a given Variant ID.
func (s *Store) GetIdentityIDForVariant(ctx context.Context, variantID int) (int, error) {
	var identityID int
	err := s.pool.QueryRow(ctx, "SELECT identity_id FROM variants WHERE id = $1", variantID).Scan(&identityID)
	if err == pgx.ErrNoRows {
		return 0, fmt.Errorf("variant %d not found", variantID)
	}
	return identityID, err
}

// SetVariantLabel links an existing variant to an Identity (creating it if needed) and names the variant.
func (s *Store) SetVariantLabel(ctx context.Context, variantID int, identityName string, variantName string) error {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)

	// Upsert Identity
	var identityID int
	err = tx.QueryRow(ctx, "INSERT INTO identities (name) VALUES ($1) ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name RETURNING id", identityName).Scan(&identityID)
	if err != nil {
		return fmt.Errorf("failed to get/create identity: %w", err)
	}

	// Update Variant
	_, err = tx.Exec(ctx, "UPDATE variants SET identity_id = $1, name = $2 WHERE id = $3", identityID, variantName, variantID)
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
	if len(newVec) != len(oldVec) {
		return fmt.Errorf("vector dimension mismatch: expected %d, got %d", len(oldVec), len(newVec))
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

// RenameIdentity renames an Identity directly by its ID.
func (s *Store) RenameIdentity(ctx context.Context, identityID int, newName string) error {
	_, err := s.pool.Exec(ctx, "UPDATE identities SET name = $1 WHERE id = $2", newName, identityID)
	return err
}

// RenameVariant updates the name of a specific variant.
func (s *Store) RenameVariant(ctx context.Context, variantID int, newName string) error {
	_, err := s.pool.Exec(ctx, "UPDATE variants SET name = $1 WHERE id = $2", newName, variantID)
	return err
}

// DeleteIdentity removes an identity from the database.
// Used for cleaning up "ghost" identities that were created but filtered out as blips.
func (s *Store) DeleteIdentity(ctx context.Context, id int) error {
	_, err := s.pool.Exec(ctx, "DELETE FROM variants WHERE id = $1", id)
	return err
}

// ApplyVariantDelta updates an identity by adding (or subtracting) a sum vector and frame count.
// This is used for both Commits (positive delta) and Rollbacks (negative delta).
func (s *Store) ApplyVariantDelta(ctx context.Context, variantID int, sumDelta []float64, countDelta int) error {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)

	if err := s.applyVariantDeltaTx(ctx, tx, variantID, sumDelta, countDelta); err != nil {
		return err
	}

	return tx.Commit(ctx)
}

// applyVariantDeltaTx performs the actual logic inside an existing transaction.
func (s *Store) applyVariantDeltaTx(ctx context.Context, tx pgx.Tx, variantID int, sumDelta []float64, countDelta int) error {
	var oldVec []float32
	var oldCount int
	var identityID int
	err = tx.QueryRow(ctx, "SELECT embedding::real[], face_count, identity_id FROM variants WHERE id = $1 FOR UPDATE", variantID).Scan(&oldVec, &oldCount, &identityID)
	if err != nil {
		return fmt.Errorf("failed to fetch variant %d: %w", variantID, err)
	}
	if len(sumDelta) != len(oldVec) {
		return fmt.Errorf("vector dimension mismatch: expected %d, got %d", len(oldVec), len(sumDelta))
	}

	newCount := oldCount + countDelta

	if newCount <= 0 {
		// Delete the variant if it becomes empty
		if _, err := tx.Exec(ctx, "DELETE FROM variants WHERE id = $1", variantID); err != nil {
			return err
		}

		// Clean up: Check if the parent identity is now empty (orphaned) and delete it if so.
		// This prevents "ghost" identities from accumulating after rollbacks.
		var remainingVariants int
		err = tx.QueryRow(ctx, "SELECT COUNT(*) FROM variants WHERE identity_id = $1", identityID).Scan(&remainingVariants)
		if err != nil {
			return err
		}

		if remainingVariants == 0 {
			if _, err := tx.Exec(ctx, "DELETE FROM identities WHERE id = $1", identityID); err != nil {
				return err
			}
		}

		return nil
	}

	// Calculate new mean
	// NewMean = (OldMean * OldCount + DeltaSum) / NewCount
	finalVec := make([]float32, len(oldVec))
	for i := range oldVec {
		currentSum := float64(oldVec[i]) * float64(oldCount)
		newMean := (currentSum + sumDelta[i]) / float64(newCount)
		finalVec[i] = float32(newMean)
	}

	_, err = tx.Exec(ctx, "UPDATE variants SET embedding = $1::real[]::vector, face_count = $2 WHERE id = $3", finalVec, newCount, variantID)
	return err
}

// LedgerEntry represents a record in the transactional ledger.
type LedgerEntry struct {
	CommitID   string
	TrackID    string
	VariantID  int
	AddedSum   []float64
	AddedCount int
}

// InsertLedgerEntry records a change in the ledger.
func (s *Store) InsertLedgerEntry(ctx context.Context, entry LedgerEntry) error {
	vec32 := make([]float32, len(entry.AddedSum))
	for i, v := range entry.AddedSum {
		vec32[i] = float32(v)
	}
	_, err := s.pool.Exec(ctx, `
		INSERT INTO ledger_entries (commit_id, track_id, variant_id, added_sum, added_count)
		VALUES ($1, $2, $3, $4::real[]::vector, $5)
	`, entry.CommitID, entry.TrackID, entry.VariantID, vec32, entry.AddedCount)
	return err
}

// --- Atomic Commit Actions (Action + Ledger) ---

// CommitMerge performs a variant merge and ledger insert atomically.
func (s *Store) CommitMerge(ctx context.Context, variantID int, addedSum []float64, addedCount int, entry LedgerEntry) error {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)

	// 1. Apply Math
	if err := s.applyVariantDeltaTx(ctx, tx, variantID, addedSum, addedCount); err != nil {
		return err
	}

	// 2. Write Ledger (Reusing logic but with direct Tx execution for safety)
	entry.VariantID = variantID // Ensure consistency
	vec32 := make([]float32, len(entry.AddedSum))
	for i, v := range entry.AddedSum {
		vec32[i] = float32(v)
	}
	_, err = tx.Exec(ctx, `
		INSERT INTO ledger_entries (commit_id, track_id, variant_id, added_sum, added_count)
		VALUES ($1, $2, $3, $4::real[]::vector, $5)
	`, entry.CommitID, entry.TrackID, entry.VariantID, vec32, entry.AddedCount)
	if err != nil {
		return err
	}

	return tx.Commit(ctx)
}

// CommitNewIdentity creates an identity, handles optional renaming, and writes to the ledger atomically.
func (s *Store) CommitNewIdentity(ctx context.Context, vec []float64, count int, entry LedgerEntry, nameOverride string) error {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)

	// 1. Create Identity & Variant (Logic copied from CreateIdentity but using tx)
	vec32 := make([]float32, len(vec))
	for i, v := range vec {
		vec32[i] = float32(v)
	}

	var identityID, variantID int
	if err := tx.QueryRow(ctx, "INSERT INTO identities DEFAULT VALUES RETURNING id").Scan(&identityID); err != nil {
		return err
	}
	if err := tx.QueryRow(ctx, "INSERT INTO variants (embedding, face_count, identity_id, name) VALUES ($1::real[]::vector, $2, $3, 'Default') RETURNING id", vec32, count, identityID).Scan(&variantID); err != nil {
		return err
	}

	// 2. Optional Rename
	if nameOverride != "" {
		if _, err := tx.Exec(ctx, "UPDATE identities SET name = $1 WHERE id = $2", nameOverride, identityID); err != nil {
			// We fail the commit if the name is taken, rather than ignoring it, to maintain consistency.
			// Callers should handle unique constraint errors if needed.
			return fmt.Errorf("failed to apply name '%s': %w", nameOverride, err)
		}
	}

	// 3. Write Ledger
	entry.VariantID = variantID
	// Ledger 'AddedSum' is already in the entry struct passed in
	ledgerVec32 := make([]float32, len(entry.AddedSum))
	for i, v := range entry.AddedSum {
		ledgerVec32[i] = float32(v)
	}
	_, err = tx.Exec(ctx, `
		INSERT INTO ledger_entries (commit_id, track_id, variant_id, added_sum, added_count)
		VALUES ($1, $2, $3, $4::real[]::vector, $5)
	`, entry.CommitID, entry.TrackID, entry.VariantID, ledgerVec32, entry.AddedCount)
	if err != nil {
		return err
	}

	return tx.Commit(ctx)
}

// GetLedgerEntries retrieves all entries for a specific commit.
func (s *Store) GetLedgerEntries(ctx context.Context, commitID string) ([]LedgerEntry, error) {
	// Fix: ORDER BY id DESC now works correctly because ID is BIGSERIAL (Chronological)
	rows, err := s.pool.Query(ctx, "SELECT track_id, variant_id, added_sum::real[], added_count FROM ledger_entries WHERE commit_id = $1 ORDER BY id DESC", commitID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var entries []LedgerEntry
	for rows.Next() {
		var e LedgerEntry
		var vec32 []float32
		if err := rows.Scan(&e.TrackID, &e.VariantID, &vec32, &e.AddedCount); err != nil {
			return nil, err
		}
		e.CommitID = commitID
		e.AddedSum = make([]float64, len(vec32))
		for i, v := range vec32 {
			e.AddedSum[i] = float64(v)
		}
		entries = append(entries, e)
	}
	return entries, nil
}

// RevertCommit performs an atomic rollback of a commit.
func (s *Store) RevertCommit(ctx context.Context, commitID string) error {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)

	// 1. Lock and Check Status
	var status string
	err = tx.QueryRow(ctx, "SELECT status FROM commits WHERE id = $1 FOR UPDATE", commitID).Scan(&status)
	if err != nil {
		if err == pgx.ErrNoRows {
			return fmt.Errorf("commit %s not found", commitID)
		}
		return err
	}

	if status == "rolled_back" {
		return fmt.Errorf("commit already rolled back")
	}
	if status == "processing" {
		return fmt.Errorf("commit is still processing (or crashed); mark as 'failed' manually before rolling back")
	}

	// 2. Fetch Ledger Entries (We can fetch inside the transaction for consistency)
	// Note: We use the *Tx* query here, reusing the logic of GetLedgerEntries but executing on the specific connection.
	// Ideally we'd call s.GetLedgerEntries but that uses s.pool directly.
	// For simplicity, we can fetch them into memory here using the Tx.
	rows, err := tx.Query(ctx, "SELECT track_id, variant_id, added_sum::real[], added_count FROM ledger_entries WHERE commit_id = $1 ORDER BY id DESC", commitID)
	if err != nil {
		return err
	}

	var entries []LedgerEntry
	for rows.Next() {
		var e LedgerEntry
		var vec32 []float32
		if err := rows.Scan(&e.TrackID, &e.VariantID, &vec32, &e.AddedCount); err != nil {
			rows.Close()
			return err
		}
		e.AddedSum = make([]float64, len(vec32))
		for i, v := range vec32 {
			e.AddedSum[i] = float64(v)
		}
		entries = append(entries, e)
	}
	rows.Close()

	if len(entries) == 0 {
		// Empty commit? Just mark it rolled back.
		_, err = tx.Exec(ctx, "UPDATE commits SET status = 'rolled_back' WHERE id = $1", commitID)
		if err != nil {
			return err
		}
		return tx.Commit(ctx)
	}

	// 3. Mark as 'rolling_back' (audit trail within the transaction)
	_, err = tx.Exec(ctx, "UPDATE commits SET status = 'rolling_back' WHERE id = $1", commitID)
	if err != nil {
		return err
	}

	// 4. Apply Reversals
	for _, entry := range entries {
		negSum := make([]float64, len(entry.AddedSum))
		for i, v := range entry.AddedSum {
			negSum[i] = -v
		}
		negCount := -entry.AddedCount

		if err := s.applyVariantDeltaTx(ctx, tx, entry.VariantID, negSum, negCount); err != nil {
			return fmt.Errorf("failed to revert variant %d: %w", entry.VariantID, err)
		}
	}

	// 5. Mark as 'rolled_back'
	_, err = tx.Exec(ctx, "UPDATE commits SET status = 'rolled_back' WHERE id = $1", commitID)
	if err != nil {
		return err
	}

	return tx.Commit(ctx)
}

// CommitNewVariant creates a variant, handles auto-naming, and writes to ledger atomically.
func (s *Store) CommitNewVariant(ctx context.Context, identityID int, vec []float64, count int, variantName string, entry LedgerEntry, autoRename bool) error {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)

	vec32 := make([]float32, len(vec))
	for i, v := range vec {
		vec32[i] = float32(v)
	}

	var variantID int
	err = tx.QueryRow(ctx,
		"INSERT INTO variants (identity_id, embedding, face_count, name) VALUES ($1, $2::real[]::vector, $3, $4) RETURNING id",
		identityID, vec32, count, variantName).Scan(&variantID)
	if err != nil {
		return err
	}

	if autoRename {
		finalName := fmt.Sprintf("Variant %d", variantID)
		if _, err := tx.Exec(ctx, "UPDATE variants SET name = $1 WHERE id = $2", finalName, variantID); err != nil {
			return err
		}
	}

	// Write Ledger
	entry.VariantID = variantID
	ledgerVec32 := make([]float32, len(entry.AddedSum))
	for i, v := range entry.AddedSum {
		ledgerVec32[i] = float32(v)
	}
	_, err = tx.Exec(ctx, `
		INSERT INTO ledger_entries (commit_id, track_id, variant_id, added_sum, added_count)
		VALUES ($1, $2, $3, $4::real[]::vector, $5)
	`, entry.CommitID, entry.TrackID, entry.VariantID, ledgerVec32, entry.AddedCount)
	if err != nil {
		return err
	}

	return tx.Commit(ctx)
}

// Reset drops all application tables to clear the database state.
// This is useful for development to force a schema refresh without migrations.
func (s *Store) Reset(ctx context.Context) error {
	_, err := s.pool.Exec(ctx, `
		DROP TABLE IF EXISTS ledger_entries CASCADE;
		DROP TABLE IF EXISTS commits CASCADE;
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
	VariantCount int // Number of variants for this identity
	CreatedAt    time.Time
}

type VariantMetadata struct {
	ID        int
	Name      string
	Count     int // Face count for this specific variant
	CreatedAt time.Time
}

func (s *Store) ListIdentities(ctx context.Context) ([]IdentityMetadata, error) {
	// List Identities
	// We LEFT JOIN so that identities with no variants yet still appear.
	// We SUM the face_count from all variants belonging to an identity.
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

func (s *Store) ListVariantsForIdentity(ctx context.Context, identityID int) ([]VariantMetadata, error) {
	query := `
		SELECT id, COALESCE(name, 'Default'), face_count, created_at
		FROM variants
		WHERE identity_id = $1
		ORDER BY id ASC`
	rows, err := s.pool.Query(ctx, query, identityID)
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

// GetIntervalsForIdentity retrieves all appearances for an *entire* identity, across all its variants.
func (s *Store) GetIntervalsForIdentity(ctx context.Context, identityID int) ([]IntervalResult, error) {
	query := `
		SELECT f.video_id, v.path, f.start_time, f.end_time
		FROM face_intervals f
		JOIN variants va ON f.variant_id = va.id
		JOIN video_metadata v ON f.video_id = v.id
		WHERE va.identity_id = $1
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
	return results, rows.Err()
}
