package cmd

import (
	"context"
	"fmt"

	"github.com/andresmejia3/sentinel/internal/utils"
	"github.com/spf13/cobra"
)

var rollbackCmd = &cobra.Command{
	Use:   "rollback <commit_id>",
	Short: "Undo a specific commit using the transaction ledger",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		cmd.SilenceUsage = true
		return runRollback(cmd.Context(), args[0])
	},
}

func init() {
	rootCmd.AddCommand(rollbackCmd)
}

func runRollback(ctx context.Context, commitID string) error {
	fmt.Printf("↺ Rolling back commit: %s\n", commitID)

	entries, err := DB.GetLedgerEntries(ctx, commitID)
	if err != nil {
		utils.ShowError("Failed to fetch ledger entries", err, nil)
		return err
	}

	if len(entries) == 0 {
		fmt.Println("⚠️  No ledger entries found for this commit ID.")
		return nil
	}

	for _, entry := range entries {
		// Math: Subtraction is adding a negative sum
		negSum := make([]float64, len(entry.AddedSum))
		for i, v := range entry.AddedSum {
			negSum[i] = -v
		}
		negCount := -entry.AddedCount

		if err := DB.ApplyVariantDelta(ctx, entry.VariantID, negSum, negCount); err != nil {
			return fmt.Errorf("failed to rollback track %s (Variant %d): %w", entry.TrackID, entry.VariantID, err)
		}
	}

	fmt.Printf("✅ Rollback Successful. Reverted %d entries.\n", len(entries))
	return nil
}
