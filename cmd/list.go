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
	Short: "List all master identities in the database",
	RunE: func(cmd *cobra.Command, args []string) error {
		cmd.SilenceUsage = true
		return runList()
	},
}

func init() {
	rootCmd.AddCommand(listCmd)
	listCmd.AddCommand(listVariantsCmd)
}

var listVariantsCmd = &cobra.Command{
	Use:   "variants <master_id>",
	Short: "List all variants for a specific master identity",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		cmd.SilenceUsage = true
		id, err := strconv.Atoi(args[0])
		if err != nil {
			utils.ShowError("Invalid master ID", err, nil)
			return err
		}
		return runListVariants(id)
	},
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
	fmt.Fprintln(w, "ID\tNAME\tFACE COUNT\tCREATED")
	fmt.Fprintln(w, "--\t----\t----------\t-------")

	for _, id := range identities {
		fmt.Fprintf(w, "%d\t%s\t%d\t%s\n", id.ID, id.Name, id.Count, id.CreatedAt.Local().Format("2006-01-02 15:04"))
	}
	w.Flush()
	return nil
}

func runListVariants(masterID int) error {
	ctx := context.Background()
	variants, err := DB.ListVariantsForIdentity(ctx, masterID)
	if err != nil {
		utils.ShowError("Failed to list variants", err, nil)
		return err
	}

	if len(variants) == 0 {
		// Check if the master ID is valid to provide a better error message.
		exists, err := DB.MasterIdentityExists(ctx, masterID)
		if err != nil {
			utils.ShowError("Failed to verify master identity", err, nil)
			return err
		}
		if !exists {
			fmt.Printf("Error: Master identity %d not found.\n", masterID)
		} else {
			fmt.Printf("No variants found for master identity %d.\n", masterID)
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
