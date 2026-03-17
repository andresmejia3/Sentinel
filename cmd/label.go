package cmd

import (
	"context"
	"fmt"
	"strconv"

	"github.com/spf13/cobra"
)

var labelCmd = &cobra.Command{
	Use:   "label",
	Short: "Label identities or variants",
}

var labelIdentityCmd = &cobra.Command{
	Use:   "identity <identity_id> <new_name>",
	Short: "Assign a new name to an identity directly",
	Args:  cobra.ExactArgs(2),
	RunE: func(cmd *cobra.Command, args []string) error {
		cmd.SilenceUsage = true
		id, err := strconv.Atoi(args[0])
		if err != nil {
			return fmt.Errorf("invalid identity ID: %q is not an integer", args[0])
		}
		name := args[1]
		return runLabelIdentity(id, name)
	},
}

var labelVariantCmd = &cobra.Command{
	Use:   "variant <variant_id> <identity_name> <variant_name>",
	Short: "Link a detected variant to an identity and name it (e.g. 'Monica' 'Glasses')",
	Args:  cobra.ExactArgs(3),
	RunE: func(cmd *cobra.Command, args []string) error {
		cmd.SilenceUsage = true
		id, err := strconv.Atoi(args[0])
		if err != nil {
			return fmt.Errorf("invalid variant ID: %q is not an integer", args[0])
		}
		identityName := args[1]
		variantName := args[2]
		return runLabelVariant(id, identityName, variantName)
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
		return fmt.Errorf("failed to label identity: %w", err)
	}

	fmt.Printf("✅ Identity %d has been labeled as '%s'\n", id, name)
	return nil
}

func runLabelVariant(id int, identityName, variantName string) error {
	ctx := context.Background()

	if err := DB.SetVariantLabel(ctx, id, identityName, variantName); err != nil {
		return fmt.Errorf("failed to label variant: %w", err)
	}

	fmt.Printf("✅ Variant %d linked to '%s' as '%s'\n", id, identityName, variantName)
	return nil
}
