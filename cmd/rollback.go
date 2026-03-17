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

	// Atomic Rollback via Store
	if err := DB.RevertCommit(ctx, commitID); err != nil {
		// The root handler will now display the formatted error.
		return fmt.Errorf("rollback failed: %w", err)
	}

	fmt.Println("✅ Rollback Successful.")
	return nil
}
