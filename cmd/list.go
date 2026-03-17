package cmd

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"text/tabwriter"

	"github.com/andresmejia3/sentinel/internal/utils"
	"github.com/spf13/cobra"
)

var listCmd = &cobra.Command{
	Use:   "list",
	Short: "List all identities in the database",
	RunE: func(cmd *cobra.Command, args []string) error {
		cmd.SilenceUsage = true
		return runList()
	},
}

var listCommitsLimit int

var listVariantsCmd = &cobra.Command{
	Use:   "variants <identity_id>",
	Short: "List all variants for a specific identity",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		cmd.SilenceUsage = true
		id, err := strconv.Atoi(args[0])
		if err != nil {
			utils.ShowError("Invalid identity ID", err, nil)
			return err
		}
		return runListVariants(id)
	},
}

var listCommitsCmd = &cobra.Command{
	Use:   "commits",
	Short: "List transaction history (commits)",
	RunE: func(cmd *cobra.Command, args []string) error {
		cmd.SilenceUsage = true
		return runListCommits(listCommitsLimit)
	},
}

func init() {
	rootCmd.AddCommand(listCmd)
	listCmd.AddCommand(listVariantsCmd)
	listCmd.AddCommand(listCommitsCmd)

	listCommitsCmd.Flags().IntVarP(&listCommitsLimit, "n", "n", 0, "Limit number of commits to show (0 = all)")
}

func runList() error {
	ctx := context.Background()
	identities, err := DB.ListIdentities(ctx)
	if err != nil {
		utils.ShowError("Failed to list identities", err, nil)
		return err
	}

	if len(identities) == 0 {
		fmt.Println("No identities found in database.")
		return nil
	}

	w := tabwriter.NewWriter(os.Stdout, 0, 0, 3, ' ', 0)
	fmt.Fprintln(w, "ID\tNAME\tVARIANTS\tFACE COUNT\tCREATED")
	fmt.Fprintln(w, "--\t----\t--------\t----------\t-------")

	for _, id := range identities {
		fmt.Fprintf(w, "%d\t%s\t%d\t%d\t%s\n", id.ID, id.Name, id.VariantCount, id.Count, id.CreatedAt.Local().Format("2006-01-02 15:04"))
	}
	w.Flush()
	return nil
}

func runListCommits(limit int) error {
	ctx := context.Background()
	commits, err := DB.ListCommits(ctx, limit)
	if err != nil {
		utils.ShowError("Failed to list commits", err, nil)
		return err
	}

	if len(commits) == 0 {
		fmt.Println("No commits found in ledger.")
		return nil
	}

	w := tabwriter.NewWriter(os.Stdout, 0, 0, 3, ' ', 0)
	fmt.Fprintln(w, "COMMIT ID\tSTATUS\tTRACKS\tFACES ADDED\tDATE")
	fmt.Fprintln(w, "---------\t------\t------\t-----------\t----")
	for _, c := range commits {
		fmt.Fprintf(w, "%s\t%s\t%d\t%d\t%s\n", c.CommitID[:8], c.Status, c.TrackCount, c.TotalFaces, c.CreatedAt.Local().Format("2006-01-02 15:04"))
	}
	w.Flush()
	return nil
}

func runListVariants(identityID int) error {
	ctx := context.Background()
	variants, err := DB.ListVariantsForIdentity(ctx, identityID)
	if err != nil {
		utils.ShowError("Failed to list variants", err, nil)
		return err
	}

	if len(variants) == 0 {
		// Check if the identity ID is valid to provide a better error message.
		exists, err := DB.IdentityExists(ctx, identityID)
		if err != nil {
			utils.ShowError("Failed to verify identity", err, nil)
			return err
		}
		if !exists {
			fmt.Printf("Error: Identity %d not found.\n", identityID)
		} else {
			fmt.Printf("No variants found for identity %d.\n", identityID)
		}
		return nil
	}

	w := tabwriter.NewWriter(os.Stdout, 0, 0, 3, ' ', 0)
	fmt.Fprintf(w, "VARIANT ID\tNAME\tFACE COUNT\tCREATED\n")
	fmt.Fprintf(w, "----------\t----\t----------\t-------\n")

	for _, v := range variants {
		fmt.Fprintf(w, "%d\t%s\t%d\t%s\n", v.ID, v.Name, v.Count, v.CreatedAt.Local().Format("2006-01-02 15:04"))
	}
	w.Flush()
	return nil
}
