package cmd

import (
	"context"
	"fmt"
	"os"
	"regexp"
	"strconv"

	"github.com/andresmejia3/sentinel/internal/store"
	"github.com/andresmejia3/sentinel/internal/utils"
	"github.com/google/uuid"
	"github.com/spf13/cobra"
	"gopkg.in/yaml.v3"
)

var commitCmd = &cobra.Command{
	Use:   "commit <staging.yaml>",
	Short: "Commit staged changes to the database with transaction ledger",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		cmd.SilenceUsage = true
		return runCommit(cmd.Context(), args[0])
	},
}

func init() {
	rootCmd.AddCommand(commitCmd)
}

// Use a named return parameter `(err error)` so that the defer block can reliably
// inspect the final error state of the function before it returns.
func runCommit(ctx context.Context, stagingPath string) (err error) {
	var data []byte
	data, err = os.ReadFile(stagingPath)
	if err != nil {
		utils.ShowError("Failed to read staging file", err, nil)
		return err
	}

	var items []StagingItem
	if err := yaml.Unmarshal(data, &items); err != nil {
		utils.ShowError("Failed to parse staging YAML", err, nil)
		return err
	}

	// 1. Strict Pre-Validation (Fixing Bug #4: Trojan Horse Vector)
	// Ensure the entire file is valid before we start ANY DB operations.
	// This prevents partial commits due to bad data.
	fmt.Println("🔍 Validating staging file...")
	for i, item := range items {
		if item.Action == "discard" || item.Action == "" {
			continue
		}
		// Check Vector Dimensions
		if len(item.InternalVector) != 512 {
			return fmt.Errorf("validation failed on item %d (%s): vector has %d dimensions, expected 512", i, item.TrackID, len(item.InternalVector))
		}
		// Check Action validity
		if item.Action != "merge" && item.Action != "new_identity" && item.Action != "new_variant" {
			return fmt.Errorf("validation failed on item %d (%s): invalid action '%s'", i, item.TrackID, item.Action)
		}
		if item.InternalCount <= 0 {
			return fmt.Errorf("validation failed on item %d (%s): internal_count must be positive, got %d", i, item.TrackID, item.InternalCount)
		}
	}

	commitID := uuid.New().String()
	fmt.Printf("🚀 Starting Commit Batch: %s\n", commitID)

	// Register Commit (Fixing Bug #5: Atomicity Gap / Tracking)
	if err = DB.CreateCommit(ctx, commitID); err != nil {
		utils.ShowError("Failed to register commit", err, nil)
		return err
	}

	// Ensure status is updated on exit
	defer func() {
		if err != nil {
			fmt.Println("\n❌ Batch failed. Marking commit as 'failed'.")
			_ = DB.MarkCommitStatus(ctx, commitID, "failed")
		}
	}()

	successCount := 0
	skipCount := 0

	for _, item := range items {
		if item.Action == "discard" || item.Action == "" {
			skipCount++
			continue
		}

		// Calculate Added Sum for Ledger (Mean * Count)
		addedSum := make([]float64, len(item.InternalVector))
		for i, v := range item.InternalVector {
			addedSum[i] = v * float64(item.InternalCount)
		}

		ledgerEntry := store.LedgerEntry{
			CommitID:   commitID,
			TrackID:    item.TrackID,
			AddedSum:   addedSum,
			AddedCount: item.InternalCount,
		}

		switch item.Action {
		case "merge":
			// Respect the staging file names
			id, err := resolveIdentityID(ctx, item.SuggestedIdentity)
			if err != nil {
				return fmt.Errorf("error resolving identity '%s' for track %s: %w", item.SuggestedIdentity, item.TrackID, err)
			}
			if id == 0 {
				return fmt.Errorf("identity '%s' not found for track %s (cannot merge)", item.SuggestedIdentity, item.TrackID)
			}

			// Enforce variant name for merges (User must explicitly select target)
			targetVariant := item.SuggestedVariant
			if targetVariant == "" {
				return fmt.Errorf("variant name is required for 'merge' action (track %s)", item.TrackID)
			}

			vid, err := DB.GetVariantID(ctx, id, targetVariant)
			if err != nil {
				return fmt.Errorf("error resolving variant '%s' for track %s: %w", targetVariant, item.TrackID, err)
			}
			if vid == 0 {
				// If variant doesn't exist, we can't merge math into it.
				// User likely meant "new_variant" or mistyped the variant name.
				return fmt.Errorf("variant '%s' does not exist for identity '%s' (track %s). Did you mean 'new_variant'?", targetVariant, item.SuggestedIdentity, item.TrackID)
			}

			if err := DB.CommitMerge(ctx, vid, addedSum, item.InternalCount, ledgerEntry); err != nil {
				return fmt.Errorf("failed to commit merge for %s: %w", item.TrackID, err)
			}

		case "new_identity":
			nameOverride := ""
			if item.SuggestedIdentity != "" && !isSystemName(item.SuggestedIdentity) {
				nameOverride = item.SuggestedIdentity
			}
			if err := DB.CommitNewIdentity(ctx, item.InternalVector, item.InternalCount, ledgerEntry, nameOverride); err != nil {
				return fmt.Errorf("failed to commit new identity for %s: %w", item.TrackID, err)
			}

		case "new_variant":
			id, err := resolveIdentityID(ctx, item.SuggestedIdentity)
			if err != nil {
				return fmt.Errorf("error resolving identity '%s' for track %s: %w", item.SuggestedIdentity, item.TrackID, err)
			}
			if id == 0 {
				return fmt.Errorf("identity '%s' not found for track %s (cannot create variant)", item.SuggestedIdentity, item.TrackID)
			}

			targetVariant := item.SuggestedVariant
			isAutoName := false

			if targetVariant == "" {
				// Use a placeholder name to satisfy UNIQUE constraints during insertion,
				// then rename it to "Variant <ID>" once we know the ID.
				targetVariant = "Pending_" + uuid.New().String()
				isAutoName = true
			} else {
				// Check uniqueness for user-provided names
				existingVid, _ := DB.GetVariantID(ctx, id, targetVariant)
				if existingVid != 0 {
					return fmt.Errorf("variant '%s' already exists for identity '%s'. Use 'merge' to add to it", targetVariant, item.SuggestedIdentity)
				}
			}

			if err := DB.CommitNewVariant(ctx, id, item.InternalVector, item.InternalCount, targetVariant, ledgerEntry, isAutoName); err != nil {
				return fmt.Errorf("failed to commit new variant for %s: %w", item.TrackID, err)
			}

		default:
			return fmt.Errorf("unknown action '%s' for track %s (valid: merge, new_identity, new_variant, discard)", item.Action, item.TrackID)
		}

		successCount++
	}

	fmt.Println("---------------------------------------------------------")
	fmt.Printf("✅ Commit Batch Complete\n")
	fmt.Printf("🆔 Commit ID: %s  <-- Save this for rollback!\n", commitID)
	fmt.Printf("📊 Processed: %d  |  Skipped: %d\n", successCount, skipCount)
	fmt.Println("---------------------------------------------------------")

	// Explicitly mark as active now that we are done
	if err = DB.MarkCommitStatus(ctx, commitID, "active"); err != nil {
		return fmt.Errorf("failed to finalize commit status: %w", err)
	}

	return nil
}

// Helper to handle "Identity 105" vs "John Doe"
func resolveIdentityID(ctx context.Context, name string) (int, error) {
	// 1. Try exact name lookup
	id, err := DB.GetIdentityIDByName(ctx, name)
	if err != nil {
		return 0, err
	}
	if id != 0 {
		return id, nil
	}

	// 2. Fallback: Check if it looks like "Identity <ID>" (System generated)
	// The DB might have NULL name, so we parse the integer ID directly.
	re := regexp.MustCompile(`^Identity (\d+)$`)
	matches := re.FindStringSubmatch(name)
	if len(matches) == 2 {
		parsedID, _ := strconv.Atoi(matches[1])
		return parsedID, nil
	}

	return 0, nil
}

func isSystemName(name string) bool {
	match, _ := regexp.MatchString(`^Identity \d+.*`, name)
	return match
}
