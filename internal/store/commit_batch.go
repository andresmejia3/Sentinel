package store

import (
	"context"
	"fmt"
	"regexp"
	"strconv"

	"github.com/jackc/pgx/v5"
)

// CommitAction is a single reviewed staging action to be applied atomically.
type CommitAction struct {
	TrackID       string
	Action        string
	IdentityName  string
	VariantName   string
	Vector        []float64
	Count         int
	TargetTrackID string
}

type CommitInterval struct {
	TrackID   string
	Start     float64
	End       float64
	FaceCount int
}

var (
	identityNamePattern = regexp.MustCompile(`(?i)^identity (\d+)$`)
)

// ApplyCommitBatch applies an entire staging batch inside one transaction.
func (s *Store) ApplyCommitBatch(ctx context.Context, commitID string, actions []CommitAction, videoID, videoPath string, intervals []CommitInterval) error {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)

	if _, err := tx.Exec(ctx, "INSERT INTO commits (id, status) VALUES ($1, 'processing')", commitID); err != nil {
		return fmt.Errorf("failed to register commit: %w", err)
	}

	if videoID != "" {
		if err := snapshotVideoStateTx(ctx, tx, commitID, videoID); err != nil {
			return err
		}
		if videoPath == "" {
			return fmt.Errorf("video path is required when committing review intervals")
		}
		if _, err := tx.Exec(ctx, `
			INSERT INTO video_metadata (id, path, indexed_at)
			VALUES ($1, $2, NOW())
			ON CONFLICT (id) DO UPDATE SET indexed_at = NOW(), path = EXCLUDED.path
		`, videoID, videoPath); err != nil {
			return fmt.Errorf("failed to register video metadata: %w", err)
		}
	}

	variantByTrack := make(map[string]int, len(actions))
	for _, action := range orderCommitActions(actions) {
		variantID, err := applyCommitActionTx(ctx, tx, commitID, action, variantByTrack)
		if err != nil {
			return err
		}
		variantByTrack[action.TrackID] = variantID
	}

	if videoID != "" {
		if _, err := tx.Exec(ctx, "DELETE FROM face_intervals WHERE video_id = $1", videoID); err != nil {
			return fmt.Errorf("failed to replace old intervals for %s: %w", videoID, err)
		}
		if len(intervals) > 0 {
			batch := &pgx.Batch{}
			for _, interval := range intervals {
				variantID, ok := variantByTrack[interval.TrackID]
				if !ok {
					return fmt.Errorf("missing committed variant for track %s", interval.TrackID)
				}
				batch.Queue(
					`INSERT INTO face_intervals (video_id, start_time, end_time, face_count, variant_id) VALUES ($1, $2, $3, $4, $5)`,
					videoID,
					interval.Start,
					interval.End,
					interval.FaceCount,
					variantID,
				)
			}

			br := tx.SendBatch(ctx, batch)
			for i := 0; i < len(intervals); i++ {
				if _, err := br.Exec(); err != nil {
					br.Close()
					return fmt.Errorf("batch insert failed on committed interval %d: %w", i, err)
				}
			}
			if err := br.Close(); err != nil {
				return fmt.Errorf("failed to close interval batch: %w", err)
			}
		}
	}

	if _, err := tx.Exec(ctx, "UPDATE commits SET status = 'active' WHERE id = $1", commitID); err != nil {
		return fmt.Errorf("failed to finalize commit status: %w", err)
	}

	return tx.Commit(ctx)
}

func orderCommitActions(actions []CommitAction) []CommitAction {
	ordered := make([]CommitAction, 0, len(actions))
	appendPhase := func(targetAction string) {
		for _, action := range actions {
			if action.Action == targetAction {
				ordered = append(ordered, action)
			}
		}
	}

	appendPhase("new_identity")
	appendPhase("new_variant")
	appendPhase("merge")

	for _, action := range actions {
		if action.Action != "new_identity" && action.Action != "new_variant" && action.Action != "merge" {
			ordered = append(ordered, action)
		}
	}

	return ordered
}

func applyCommitActionTx(ctx context.Context, tx pgx.Tx, commitID string, action CommitAction, variantByTrack map[string]int) (int, error) {
	addedSum := make([]float64, len(action.Vector))
	for i, v := range action.Vector {
		addedSum[i] = v * float64(action.Count)
	}

	ledgerEntry := LedgerEntry{
		CommitID:   commitID,
		TrackID:    action.TrackID,
		AddedSum:   addedSum,
		AddedCount: action.Count,
	}

	switch action.Action {
	case "merge":
		if action.TargetTrackID != "" {
			variantID, ok := variantByTrack[action.TargetTrackID]
			if !ok {
				return 0, fmt.Errorf("target track %s has not created a merge target yet for track %s", action.TargetTrackID, action.TrackID)
			}
			if err := commitMergeTx(ctx, tx, variantID, addedSum, action.Count, ledgerEntry); err != nil {
				return 0, fmt.Errorf("failed to commit grouped merge for %s: %w", action.TrackID, err)
			}
			return variantID, nil
		}
		identityID, err := resolveIdentityIDTx(ctx, tx, action.IdentityName)
		if err != nil {
			return 0, fmt.Errorf("error resolving identity '%s' for track %s: %w", action.IdentityName, action.TrackID, err)
		}
		if identityID == 0 {
			return 0, fmt.Errorf("identity '%s' not found for track %s (cannot merge)", action.IdentityName, action.TrackID)
		}
		if action.VariantName == "" {
			return 0, fmt.Errorf("variant name is required for 'merge' action (track %s)", action.TrackID)
		}

		variantID, err := getVariantIDTx(ctx, tx, identityID, action.VariantName)
		if err != nil {
			return 0, fmt.Errorf("error resolving variant '%s' for track %s: %w", action.VariantName, action.TrackID, err)
		}
		if variantID == 0 {
			return 0, fmt.Errorf("variant '%s' does not exist for identity '%s' (track %s). Did you mean 'new_variant'?", action.VariantName, action.IdentityName, action.TrackID)
		}

		if err := commitMergeTx(ctx, tx, variantID, addedSum, action.Count, ledgerEntry); err != nil {
			return 0, fmt.Errorf("failed to commit merge for %s: %w", action.TrackID, err)
		}
		return variantID, nil

	case "new_identity":
		variantID, err := commitNewIdentityTx(ctx, tx, action.Vector, action.Count, ledgerEntry, action.IdentityName)
		if err != nil {
			return 0, fmt.Errorf("failed to commit new identity for %s: %w", action.TrackID, err)
		}
		return variantID, nil

	case "new_variant":
		identityID, err := resolveIdentityIDTx(ctx, tx, action.IdentityName)
		if err != nil {
			return 0, fmt.Errorf("error resolving identity '%s' for track %s: %w", action.IdentityName, action.TrackID, err)
		}
		if identityID == 0 {
			return 0, fmt.Errorf("identity '%s' not found for track %s (cannot create variant)", action.IdentityName, action.TrackID)
		}

		targetVariant := action.VariantName
		if targetVariant == "" {
			return 0, fmt.Errorf("variant name is required for 'new_variant' action (track %s)", action.TrackID)
		}
		existingVariantID, err := getVariantIDTx(ctx, tx, identityID, targetVariant)
		if err != nil {
			return 0, fmt.Errorf("error resolving variant '%s' for track %s: %w", targetVariant, action.TrackID, err)
		}
		if existingVariantID != 0 {
			return 0, fmt.Errorf("variant '%s' already exists for identity '%s'. Use 'merge' to add to it", targetVariant, action.IdentityName)
		}

		variantID, err := commitNewVariantTx(ctx, tx, identityID, action.Vector, action.Count, targetVariant, ledgerEntry, false)
		if err != nil {
			return 0, fmt.Errorf("failed to commit new variant for %s: %w", action.TrackID, err)
		}
		return variantID, nil

	default:
		return 0, fmt.Errorf("unknown action '%s' for track %s (valid: merge, new_identity, new_variant)", action.Action, action.TrackID)
	}
}

func resolveIdentityIDTx(ctx context.Context, tx pgx.Tx, name string) (int, error) {
	id, err := getIdentityIDByNameTx(ctx, tx, name)
	if err != nil {
		return 0, err
	}
	if id != 0 {
		return id, nil
	}

	matches := identityNamePattern.FindStringSubmatch(name)
	if len(matches) == 2 {
		parsedID, _ := strconv.Atoi(matches[1])
		existingID, err := getIdentityIDByIDTx(ctx, tx, parsedID)
		if err != nil {
			return 0, err
		}
		return existingID, nil
	}

	return 0, nil
}

func getIdentityIDByNameTx(ctx context.Context, tx pgx.Tx, name string) (int, error) {
	var id int
	err := tx.QueryRow(ctx, "SELECT id FROM identities WHERE LOWER(name) = LOWER($1)", name).Scan(&id)
	if err == pgx.ErrNoRows {
		return 0, nil
	}
	return id, err
}

func getIdentityIDByIDTx(ctx context.Context, tx pgx.Tx, id int) (int, error) {
	var existingID int
	err := tx.QueryRow(ctx, "SELECT id FROM identities WHERE id = $1", id).Scan(&existingID)
	if err == pgx.ErrNoRows {
		return 0, nil
	}
	return existingID, err
}

func getVariantIDTx(ctx context.Context, tx pgx.Tx, identityID int, name string) (int, error) {
	var id int
	err := tx.QueryRow(ctx, "SELECT id FROM variants WHERE identity_id = $1 AND LOWER(name) = LOWER($2)", identityID, name).Scan(&id)
	if err == pgx.ErrNoRows {
		return 0, nil
	}
	return id, err
}

func commitMergeTx(ctx context.Context, tx pgx.Tx, variantID int, addedSum []float64, addedCount int, entry LedgerEntry) error {
	if err := (&Store{}).applyVariantDeltaTx(ctx, tx, variantID, addedSum, addedCount); err != nil {
		return err
	}
	entry.VariantID = variantID
	return insertLedgerEntryTx(ctx, tx, entry)
}

func commitNewIdentityTx(ctx context.Context, tx pgx.Tx, vec []float64, count int, entry LedgerEntry, nameOverride string) (int, error) {
	vec32 := toFloat32Slice(vec)

	var identityID, variantID int
	if err := tx.QueryRow(ctx, "INSERT INTO identities DEFAULT VALUES RETURNING id").Scan(&identityID); err != nil {
		return 0, err
	}
	if err := tx.QueryRow(ctx, "INSERT INTO variants (embedding, face_count, identity_id, name) VALUES ($1::real[]::vector, $2, $3, 'Default') RETURNING id", vec32, count, identityID).Scan(&variantID); err != nil {
		return 0, err
	}

	if nameOverride != "" {
		if _, err := tx.Exec(ctx, "UPDATE identities SET name = $1 WHERE id = $2", nameOverride, identityID); err != nil {
			return 0, fmt.Errorf("failed to apply name '%s': %w", nameOverride, err)
		}
	}

	entry.VariantID = variantID
	if err := insertLedgerEntryTx(ctx, tx, entry); err != nil {
		return 0, err
	}
	return variantID, nil
}

func commitNewVariantTx(ctx context.Context, tx pgx.Tx, identityID int, vec []float64, count int, variantName string, entry LedgerEntry, autoRename bool) (int, error) {
	vec32 := toFloat32Slice(vec)

	var variantID int
	if err := tx.QueryRow(ctx, "INSERT INTO variants (identity_id, embedding, face_count, name) VALUES ($1, $2::real[]::vector, $3, $4) RETURNING id", identityID, vec32, count, variantName).Scan(&variantID); err != nil {
		return 0, err
	}

	if autoRename {
		finalName := fmt.Sprintf("Variant %d", variantID)
		if _, err := tx.Exec(ctx, "UPDATE variants SET name = $1 WHERE id = $2", finalName, variantID); err != nil {
			return 0, err
		}
	}

	entry.VariantID = variantID
	if err := insertLedgerEntryTx(ctx, tx, entry); err != nil {
		return 0, err
	}
	return variantID, nil
}

func insertLedgerEntryTx(ctx context.Context, tx pgx.Tx, entry LedgerEntry) error {
	vec32 := toFloat32Slice(entry.AddedSum)
	_, err := tx.Exec(ctx, `
		INSERT INTO ledger_entries (commit_id, track_id, variant_id, added_sum, added_count)
		VALUES ($1, $2, $3, $4::real[]::vector, $5)
	`, entry.CommitID, entry.TrackID, entry.VariantID, vec32, entry.AddedCount)
	return err
}
