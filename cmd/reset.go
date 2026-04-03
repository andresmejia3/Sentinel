package cmd

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strings"

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
	RunE: func(cmd *cobra.Command, args []string) error {
		cmd.SilenceUsage = true
		// If no flags are set, default to clearing EVERYTHING
		if !resetDB && !resetFiles && !resetDebug {
			resetDB = true
			resetFiles = true
			resetDebug = true
		}

		reader := bufio.NewReader(os.Stdin)
		confirmedReset := false

		if resetDB {
			if confirm(reader, "⚠️  Are you sure you want to DROP all database tables?") {
				confirmedReset = true
				fmt.Println("🗑️  Clearing Database...")
				if err := DB.Reset(cmd.Context()); err != nil {
					return fmt.Errorf("failed to reset database: %w", err)
				}
			}
		}

		// Determine output base directory
		outputBase := "data"
		if _, err := os.Stat("/data"); err == nil {
			outputBase = "/data"
		}

		if resetFiles {
			if confirm(reader, "⚠️  Are you sure you want to delete all output files (review files, thumbnails, and output videos)?") {
				confirmedReset = true
				fmt.Println("🗑️  Clearing Output Files (Review Files, Thumbnails, Videos)...")
				for _, path := range resetManagedFilePaths(outputBase) {
					removeDir(path)
				}
			}
		}

		if resetDebug {
			if confirm(reader, "⚠️  Are you sure you want to delete all debug frames?") {
				confirmedReset = true
				fmt.Println("🗑️  Clearing Debug Frames...")
				removeDir(filepath.Join(outputBase, "debug_frames"))
			}
		}

		fmt.Println(resetCompletionMessage(confirmedReset))
		return nil
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

func resetManagedFilePaths(outputBase string) []string {
	return []string{
		filepath.Join(outputBase, "reviews"),
		filepath.Join(outputBase, "results"),
		filepath.Join(outputBase, "output"),
	}
}

func resetCompletionMessage(confirmedReset bool) string {
	if confirmedReset {
		return "✨ System Reset Complete."
	}
	return "ℹ️  No reset actions were confirmed."
}
