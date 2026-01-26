package cmd

import (
	"context"
	"fmt"
	"strconv"

	"github.com/andresmejia3/sentinel/internal/utils"
	"github.com/spf13/cobra"
)

var labelCmd = &cobra.Command{
	Use:   "label <identity_id> <name>",
	Short: "Assign a name to a discovered identity from the last scan",
	Args:  cobra.ExactArgs(2),
	RunE: func(cmd *cobra.Command, args []string) error {
		cmd.SilenceUsage = true
		id, err := strconv.Atoi(args[0])
		if err != nil {
			utils.ShowError("Invalid identity ID", err, nil)
			return err
		}
		name := args[1]

		return runLabel(id, name)
	},
}

func init() {
	rootCmd.AddCommand(labelCmd)
}

func runLabel(id int, name string) error {
	// Database is initialized in Root PersistentPreRun
	ctx := context.Background()

	if err := DB.RenameIdentity(ctx, id, name); err != nil {
		utils.ShowError("Failed to label identity", err, nil)
		return err
	}

	fmt.Printf("✅ Identity %d labeled as '%s'\n", id, name)
	return nil
}
