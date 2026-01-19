package cmd

import (
	"context"
	"fmt"
	"os"

	"github.com/andresmejia3/sentinel/internal/store"
	"github.com/spf13/cobra"
)

// Options holds shared configuration for scan, find, and redact commands
type Options struct {
	InputPath      string
	NthFrame       int
	NumEngines     int
	GapDuration    string
	MatchThreshold float64
}

var (
	// DB is the global database connection shared by subcommands
	DB *store.Store
	// dbURL is the connection string
	dbURL string
)

// Version is the application version.
const Version = "0.0.1"

var rootCmd = &cobra.Command{
	Use:     "sentinel",
	Short:   "Biometric Video Indexing & Redaction Engine",
	Version: Version, // This enables the --version flag
	PersistentPreRunE: func(cmd *cobra.Command, args []string) error {
		// Initialize DB connection
		var err error
		// We use a background context for the connection
		DB, err = store.New(context.Background(), dbURL)
		if err != nil {
			return fmt.Errorf("failed to connect to database: %w", err)
		}
		return nil
	},
	PersistentPostRun: func(cmd *cobra.Command, args []string) {
		if DB != nil {
			DB.Close(context.Background())
		}
	},
}

func Execute() {
	// This tells Cobra not to print the version in the help text, which is cleaner.
	rootCmd.SetVersionTemplate(`{{printf "%s\n" .Version}}`)

	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func init() {
	rootCmd.PersistentFlags().StringVar(&dbURL, "db", "postgres://localhost:5432/sentinel", "PostgreSQL connection string")
}
