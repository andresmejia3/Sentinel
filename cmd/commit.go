package cmd

import (
	"context"
	"fmt"
	"os"

	"github.com/andresmejia3/sentinel/internal/store"
	"github.com/andresmejia3/sentinel/internal/utils"
	"github.com/google/uuid"
	"github.com/spf13/cobra"
	"gopkg.in/yaml.v3"
)

var commitCmd = &cobra.Command{
	Use:   "commit <staging.yaml>",
	Short: "Commit staged changes to the database with transaction ledger",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		cmd.SilenceUsage = true
		return runCommit(cmd.Context(), args[0])
	},
}

func init() {
	rootCmd.AddCommand(commitCmd)
}

func runCommit(ctx context.Context, stagingPath string) error {
	data, err := os.ReadFile(stagingPath)
	if err != nil {
		utils.ShowError("Failed to read staging file", err, nil)
		return err
	}

	var items []StagingItem
	if err := yaml.Unmarshal(data, &items); err != nil {
		utils.ShowError("Failed to parse staging YAML", err, nil)
		return err
	}

	commitID := uuid.New().String()
	fmt.Printf("🚀 Starting Commit Batch: %s\n", commitID)

	successCount := 0
	skipCount := 0

	for _, item := range items {
		if item.Action == "discard" || item.Action == "" {
			skipCount++
			continue
		}

		// Calculate Added Sum for Ledger (Mean * Count)
		addedSum := make([]float64, len(item.InternalVector))
		for i, v := range item.InternalVector {
			addedSum[i] = v * float64(item.InternalCount)
		}

		ledgerEntry := store.LedgerEntry{
			CommitID:   commitID,
			TrackID:    item.TrackID,
			AddedSum:   addedSum,
			AddedCount: item.InternalCount,
		}

		switch item.Action {
		case "merge":
			// We need to resolve the identity to a variant ID based on name
			// NOTE: In a real UI, the YAML would contain the target ID explicitly.
			// Here we assume the SuggestedIdentity/Variant names are valid keys.
			// For robust implementation, we'd lookup by name or enforce ID in YAML.
			// Simplification: We search closest match again to get the ID for "merge" target
			// effectively trusting the suggested identity string if user didn't change it.
			vid, _, _, _, err := DB.FindClosestIdentity(ctx, item.InternalVector, 2.0)
			if err != nil {
				return err
			}
			ledgerEntry.VariantID = vid

			if err := DB.ApplyVariantDelta(ctx, vid, addedSum, item.InternalCount); err != nil {
				return fmt.Errorf("failed to merge track %s: %w", item.TrackID, err)
			}

		case "new_identity":
			vid, _, err := DB.CreateIdentity(ctx, item.InternalVector, item.InternalCount)
			if err != nil {
				return fmt.Errorf("failed to create identity for track %s: %w", item.TrackID, err)
			}
			ledgerEntry.VariantID = vid
			// CreateIdentity sets the initial vector, so we don't need ApplyVariantDelta here.
			// But we DO record it in the ledger so we can undo it (via deletion).
			// The ledger logic for rollback needs to know if it was a creation or update.
			// However, our rollback logic is simple math: Subtract sum/count.
			// If count hits 0, delete. This works perfectly for new_identity too.

		case "new_variant":
			// Assuming "new_variant" means attaching to the SuggestedIdentity as a new cluster
			// First find Identity ID
			vid, identityID, _, _, err := DB.FindClosestIdentity(ctx, item.InternalVector, 2.0)
			if err != nil {
				return err
			}
			_ = vid // We don't merge into this variant, we use its identity

			// Create new variant under IdentityID
			// Since we don't have a direct CreateVariant in Store yet, we can simulate or add it
			// For MVP: Treat as new identity but you'd realistically link it.
			// Falling back to new_identity logic for code brevity unless CreateVariant is added.
			vid, _, err = DB.CreateIdentity(ctx, item.InternalVector, item.InternalCount)
			if err := DB.SetVariantLabel(ctx, vid, item.SuggestedIdentity, "New Cluster"); err != nil {
				return err
			}
			ledgerEntry.VariantID = vid
		}

		if err := DB.InsertLedgerEntry(ctx, ledgerEntry); err != nil {
			return fmt.Errorf("failed to write ledger for track %s: %w", item.TrackID, err)
		}
		successCount++
	}

	fmt.Println("---------------------------------------------------------")
	fmt.Printf("✅ Commit Batch Complete\n")
	fmt.Printf("🆔 Commit ID: %s  <-- Save this for rollback!\n", commitID)
	fmt.Printf("📊 Processed: %d  |  Skipped: %d\n", successCount, skipCount)
	fmt.Println("---------------------------------------------------------")
	return nil
}
