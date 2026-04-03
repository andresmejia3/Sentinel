package cmd

import (
	"path/filepath"
	"testing"
)

func TestResetManagedFilePathsIncludesReviews(t *testing.T) {
	t.Parallel()

	got := resetManagedFilePaths("data")
	want := []string{
		filepath.Join("data", "reviews"),
		filepath.Join("data", "results"),
		filepath.Join("data", "output"),
	}

	if len(got) != len(want) {
		t.Fatalf("resetManagedFilePaths returned %d paths, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("resetManagedFilePaths[%d] = %q, want %q", i, got[i], want[i])
		}
	}
}

func TestResetCompletionMessage(t *testing.T) {
	t.Parallel()

	if got := resetCompletionMessage(true); got != "✨ System Reset Complete." {
		t.Fatalf("resetCompletionMessage(true) = %q", got)
	}
	if got := resetCompletionMessage(false); got != "ℹ️  No reset actions were confirmed." {
		t.Fatalf("resetCompletionMessage(false) = %q", got)
	}
}
