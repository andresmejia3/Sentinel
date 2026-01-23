package cmd

import (
	"context"
	"fmt"
	"os"
	"text/tabwriter"

	"github.com/andresmejia3/sentinel/internal/utils"
	"github.com/spf13/cobra"
)

var listCmd = &cobra.Command{
	Use:   "list",
	Short: "List all known identities in the database",
	RunE: func(cmd *cobra.Command, args []string) error {
		cmd.SilenceUsage = true
		return runList()
	},
}

func init() {
	rootCmd.AddCommand(listCmd)
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
		name := id.Name
		if name == "" {
			name = fmt.Sprintf("Identity %d", id.ID)
		}
		fmt.Fprintf(w, "%d\t%s\t%d\t%s\n", id.ID, name, id.Count, id.CreatedAt.Local().Format("2006-01-02 15:04"))
	}
	w.Flush()
	return nil
}
