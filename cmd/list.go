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
	Run: func(cmd *cobra.Command, args []string) {
		runList()
	},
}

func init() {
	rootCmd.AddCommand(listCmd)
}

func runList() {
	ctx := context.Background()
	identities, err := DB.ListIdentities(ctx)
	if err != nil {
		utils.Die("Failed to list identities", err, nil)
	}

	if len(identities) == 0 {
		fmt.Println("No identities found in database.")
		return
	}

	w := tabwriter.NewWriter(os.Stdout, 0, 0, 3, ' ', 0)
	fmt.Fprintln(w, "ID\tNAME\tFACE COUNT\tCREATED")
	fmt.Fprintln(w, "--\t----\t----------\t-------")

	for _, id := range identities {
		fmt.Fprintf(w, "%d\t%s\t%d\t%s\n", id.ID, id.Name, id.Count, id.CreatedAt.Local().Format("2006-01-02 15:04"))
	}
	w.Flush()
}
