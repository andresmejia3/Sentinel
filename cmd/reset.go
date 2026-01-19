package cmd

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"github.com/andresmejia3/sentinel/internal/utils"
	"github.com/spf13/cobra"
)

var (
	resetDB    bool
	resetFiles bool
	resetDebug bool
)

var resetCmd = &cobra.Command{
	Use:   "reset",
	Short: "Reset system state (Database, Thumbnails, Debug Frames)",
	Long:  "Clears all data. By default, it resets everything. Use flags to clear specific components.",
	Run: func(cmd *cobra.Command, args []string) {
		// If no flags are set, default to clearing EVERYTHING
		if !resetDB && !resetFiles && !resetDebug {
			resetDB = true
			resetFiles = true
			resetDebug = true
		}

		reader := bufio.NewReader(os.Stdin)

		if resetDB {
			if confirm(reader, "⚠️  Are you sure you want to DROP all database tables?") {
				fmt.Println("🗑️  Clearing Database...")
				if err := DB.Reset(cmd.Context()); err != nil {
					utils.Die("Failed to reset database", err, nil)
				}
			}
		}

		if resetFiles {
			if confirm(reader, "⚠️  Are you sure you want to delete all thumbnails and output videos?") {
				fmt.Println("🗑️  Clearing Output Files (Thumbnails, Videos)...")
				removeDir("/data/thumbnails")
				removeDir("/data/output")
			}
		}

		if resetDebug {
			if confirm(reader, "⚠️  Are you sure you want to delete all debug frames?") {
				fmt.Println("🗑️  Clearing Debug Frames...")
				removeDir("/data/debug_frames")
			}
		}

		fmt.Println("✨ System Reset Complete.")
	},
}

func init() {
	resetCmd.Flags().BoolVar(&resetDB, "db", false, "Clear PostgreSQL database")
	resetCmd.Flags().BoolVar(&resetFiles, "files", false, "Clear generated files (thumbnails, outputs)")
	resetCmd.Flags().BoolVar(&resetDebug, "debug", false, "Clear debug frames")
	rootCmd.AddCommand(resetCmd)
}

func confirm(r *bufio.Reader, prompt string) bool {
	fmt.Printf("%s [y/N]: ", prompt)
	res, _ := r.ReadString('\n')
	res = strings.TrimSpace(strings.ToLower(res))
	return res == "y" || res == "yes"
}

func removeDir(path string) {
	if err := os.RemoveAll(path); err != nil {
		fmt.Fprintf(os.Stderr, "⚠️  Failed to remove %s: %v\n", path, err)
	}
}
