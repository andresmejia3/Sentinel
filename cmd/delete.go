package cmd

import (
	"bufio"
	"fmt"
	"os"
	"strconv"

	"github.com/spf13/cobra"
)

var deleteYes bool

var deleteCmd = &cobra.Command{
	Use:   "delete",
	Short: "Delete identities or variants",
}

var deleteIdentityCmd = &cobra.Command{
	Use:   "identity <identity_id>",
	Short: "Delete an identity, all of its variants, and linked intervals",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		cmd.SilenceUsage = true
		id, err := strconv.Atoi(args[0])
		if err != nil {
			return fmt.Errorf("invalid identity ID: %q is not an integer", args[0])
		}
		return runDeleteIdentity(cmd, id)
	},
}

var deleteVariantCmd = &cobra.Command{
	Use:   "variant <variant_id>",
	Short: "Delete a variant and its linked intervals",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		cmd.SilenceUsage = true
		id, err := strconv.Atoi(args[0])
		if err != nil {
			return fmt.Errorf("invalid variant ID: %q is not an integer", args[0])
		}
		return runDeleteVariant(cmd, id)
	},
}

func init() {
	rootCmd.AddCommand(deleteCmd)
	deleteCmd.AddCommand(deleteIdentityCmd)
	deleteCmd.AddCommand(deleteVariantCmd)

	deleteIdentityCmd.Flags().BoolVarP(&deleteYes, "yes", "y", false, "Delete without confirmation prompt")
	deleteVariantCmd.Flags().BoolVarP(&deleteYes, "yes", "y", false, "Delete without confirmation prompt")
}

func runDeleteIdentity(cmd *cobra.Command, id int) error {
	if !deleteYes {
		reader := bufio.NewReader(os.Stdin)
		if !confirm(reader, fmt.Sprintf("⚠️  Delete identity %d and all of its variants and linked intervals?", id)) {
			fmt.Println("Deletion cancelled.")
			return nil
		}
	}

	if err := DB.DeleteIdentity(cmd.Context(), id); err != nil {
		return fmt.Errorf("failed to delete identity: %w", err)
	}

	fmt.Printf("🗑️  Identity %d deleted.\n", id)
	return nil
}

func runDeleteVariant(cmd *cobra.Command, id int) error {
	ctx := cmd.Context()

	identityID, err := DB.GetIdentityIDForVariant(ctx, id)
	if err != nil {
		return fmt.Errorf("failed to load variant details: %w", err)
	}

	variants, err := DB.ListVariantsForIdentity(ctx, identityID)
	if err != nil {
		return fmt.Errorf("failed to inspect sibling variants: %w", err)
	}

	lastVariant := len(variants) == 1 && variants[0].ID == id

	if !deleteYes {
		reader := bufio.NewReader(os.Stdin)
		if !confirm(reader, fmt.Sprintf("⚠️  Delete variant %d and its linked intervals?", id)) {
			fmt.Println("Deletion cancelled.")
			return nil
		}
	}

	if err := DB.DeleteVariant(ctx, id); err != nil {
		return fmt.Errorf("failed to delete variant: %w", err)
	}

	fmt.Printf("🗑️  Variant %d deleted.\n", id)

	if !lastVariant {
		return nil
	}

	if deleteYes {
		fmt.Printf("ℹ️  Identity %d now has no variants. Delete it manually with 'sentinel delete identity %d' if you want it removed too.\n", identityID, identityID)
		return nil
	}

	reader := bufio.NewReader(os.Stdin)
	if !confirm(reader, fmt.Sprintf("⚠️  Variant %d was the last variant under identity %d. Delete the now-empty identity too?", id, identityID)) {
		fmt.Printf("ℹ️  Identity %d was kept.\n", identityID)
		return nil
	}

	if err := DB.DeleteIdentity(ctx, identityID); err != nil {
		return fmt.Errorf("failed to delete now-empty identity %d: %w", identityID, err)
	}

	fmt.Printf("🗑️  Identity %d deleted.\n", identityID)
	return nil
}
