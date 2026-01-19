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
	Run: func(cmd *cobra.Command, args []string) {
		id, err := strconv.Atoi(args[0])
		if err != nil {
			utils.Die("Invalid identity ID", err, nil)
		}
		name := args[1]

		runLabel(id, name)
	},
}

func init() {
	rootCmd.AddCommand(labelCmd)
}

func runLabel(id int, name string) {
	// 1. Database is initialized in Root PersistentPreRun
	ctx := context.Background()

	// 2. Rename the existing identity
	if err := DB.RenameIdentity(ctx, id, name); err != nil {
		utils.Die("Failed to label identity", err, nil)
	}

	fmt.Printf("✅ Identity %d labeled as '%s'\n", id, name)
}
