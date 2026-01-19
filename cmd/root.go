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
	InputPath        string
	NthFrame         int
	NumEngines       int
	GracePeriod      string
	Linger           string
	MatchThreshold   float64
	DisableSafetyNet bool
	BlurStrength     int
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
		// If no flag was provided, try to build the connection string from the environment
		if dbURL == "" {
			if host := os.Getenv("POSTGRES_HOST"); host != "" {
				user := os.Getenv("POSTGRES_USER")
				pass := os.Getenv("POSTGRES_PASSWORD")
				name := os.Getenv("POSTGRES_DB")
				port := os.Getenv("POSTGRES_PORT")
				if port == "" {
					port = "5432"
				}
				dbURL = fmt.Sprintf("postgres://%s:%s@%s:%s/%s", user, pass, host, port, name)
			} else {
				// Fallback to local default if no env vars are present
				dbURL = "postgres://localhost:5432/sentinel"
			}
		}

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
	rootCmd.PersistentFlags().StringVar(&dbURL, "db", "", "PostgreSQL connection string (default: postgres://localhost:5432/sentinel)")
}
