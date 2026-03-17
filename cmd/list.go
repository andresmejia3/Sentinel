package cmd

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"text/tabwriter"

	"github.com/spf13/cobra"
)

var listCmd = &cobra.Command{
	Use:   "list",
	Short: "List all identities in the database",
	RunE: func(cmd *cobra.Command, args []string) error {
		cmd.SilenceUsage = true
		return runList(cmd.Context())
	},
}

var listLimit int
var listPage int
var listName string
var listCommitsLimit int

var listVariantsCmd = &cobra.Command{
	Use:   "variants <identity_id>",
	Short: "List all variants for a specific identity",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		cmd.SilenceUsage = true
		id, err := strconv.Atoi(args[0])
		if err != nil {
			// Return a clean error for invalid input; Cobra will print it.
			return fmt.Errorf("invalid identity ID: %q is not an integer", args[0])
		}
		return runListVariants(cmd.Context(), id)
	},
}

var listCommitsCmd = &cobra.Command{
	Use:   "commits",
	Short: "List transaction history (commits)",
	RunE: func(cmd *cobra.Command, args []string) error {
		cmd.SilenceUsage = true
		if listCommitsLimit < 0 {
			return fmt.Errorf("limit cannot be negative")
		}
		return runListCommits(cmd.Context(), listCommitsLimit)
	},
}

func init() {
	rootCmd.AddCommand(listCmd)
	listCmd.AddCommand(listVariantsCmd)
	listCmd.AddCommand(listCommitsCmd)

	listCmd.Flags().IntVarP(&listLimit, "limit", "l", 50, "Limit results per page (0 for all)")
	listCmd.Flags().IntVarP(&listPage, "page", "p", 1, "Page number")
	listCmd.Flags().StringVarP(&listName, "name", "n", "", "Filter by identity name")

	listCommitsCmd.Flags().IntVarP(&listCommitsLimit, "limit", "n", 0, "Limit number of commits to show (0 = all)")
}

func runList(ctx context.Context) error {
	if listLimit < 0 {
		return fmt.Errorf("limit cannot be negative")
	}
	if listPage < 1 {
		listPage = 1
	}
	offset := (listPage - 1) * listLimit

	identities, err := DB.ListIdentities(ctx, listLimit, offset, listName)
	if err != nil {
		return fmt.Errorf("failed to list identities: %w", err)
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

func runListCommits(ctx context.Context, limit int) error {
	commits, err := DB.ListCommits(ctx, limit)
	if err != nil {
		return fmt.Errorf("failed to list commits: %w", err)
	}

	if len(commits) == 0 {
		fmt.Println("No commits found in ledger.")
		return nil
	}

	w := tabwriter.NewWriter(os.Stdout, 0, 0, 3, ' ', 0)
	fmt.Fprintln(w, "COMMIT ID\tSTATUS\tTRACKS\tFACES ADDED\tDATE")
	fmt.Fprintln(w, "---------\t------\t------\t-----------\t----")
	for _, c := range commits {
		idDisplay := c.CommitID
		if len(c.CommitID) > 8 {
			idDisplay = c.CommitID[:8]
		}
		fmt.Fprintf(w, "%s\t%s\t%d\t%d\t%s\n", idDisplay, c.Status, c.TrackCount, c.TotalFaces, c.CreatedAt.Local().Format("2006-01-02 15:04"))
	}
	w.Flush()
	return nil
}

func runListVariants(ctx context.Context, identityID int) error {
	variants, err := DB.ListVariantsForIdentity(ctx, identityID)
	if err != nil {
		return fmt.Errorf("failed to list variants: %w", err)
	}

	if len(variants) == 0 {
		// Check if the identity ID is valid to provide a better error message.
		exists, err := DB.IdentityExists(ctx, identityID)
		if err != nil {
			return fmt.Errorf("failed to verify identity: %w", err)
		}
		if !exists {
			return fmt.Errorf("identity %d not found", identityID)
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
