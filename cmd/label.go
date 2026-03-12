package cmd

import (
	"context"
	"fmt"
	"strconv"

	"github.com/andresmejia3/sentinel/internal/utils"
	"github.com/spf13/cobra"
)

var labelCmd = &cobra.Command{
	Use:   "label",
	Short: "Label master identities or variants",
}

var labelIdentityCmd = &cobra.Command{
	Use:   "identity <master_id> <new_name>",
	Short: "Assign a new name to a master identity directly",
	Args:  cobra.ExactArgs(2),
	RunE: func(cmd *cobra.Command, args []string) error {
		cmd.SilenceUsage = true
		id, err := strconv.Atoi(args[0])
		if err != nil {
			utils.ShowError("Invalid master ID", err, nil)
			return err
		}
		name := args[1]
		return runLabelIdentity(id, name)
	},
}

var labelVariantCmd = &cobra.Command{
	Use:   "variant <variant_id> <master_name> <variant_name>",
	Short: "Link a detected variant to a master identity and name it (e.g. 'Monica' 'Glasses')",
	Args:  cobra.ExactArgs(3),
	RunE: func(cmd *cobra.Command, args []string) error {
		cmd.SilenceUsage = true
		id, err := strconv.Atoi(args[0])
		if err != nil {
			utils.ShowError("Invalid variant ID", err, nil)
			return err
		}
		masterName := args[1]
		variantName := args[2]
		return runLabelVariant(id, masterName, variantName)
	},
}

func init() {
	rootCmd.AddCommand(labelCmd)
	labelCmd.AddCommand(labelIdentityCmd)
	labelCmd.AddCommand(labelVariantCmd)
}

func runLabelIdentity(id int, name string) error {
	ctx := context.Background()

	if err := DB.RenameIdentity(ctx, id, name); err != nil {
		utils.ShowError("Failed to label master identity", err, nil)
		return err
	}

	fmt.Printf("✅ Master identity %d has been labeled as '%s'\n", id, name)
	return nil
}

func runLabelVariant(id int, masterName, variantName string) error {
	ctx := context.Background()

	if err := DB.SetVariantLabel(ctx, id, masterName, variantName); err != nil {
		utils.ShowError("Failed to label variant", err, nil)
		return err
	}

	fmt.Printf("✅ Variant %d linked to '%s' as '%s'\n", id, masterName, variantName)
	return nil
}
