package store

import "testing"

func TestRequireRowUpdated(t *testing.T) {
	t.Run("updated row succeeds", func(t *testing.T) {
		if err := requireRowUpdated(1, "identity", 7); err != nil {
			t.Fatalf("requireRowUpdated returned error: %v", err)
		}
	})

	t.Run("missing row returns not found", func(t *testing.T) {
		err := requireRowUpdated(0, "variant", 42)
		if err == nil {
			t.Fatal("expected missing row to return an error")
		}
		if got, want := err.Error(), "variant 42 not found"; got != want {
			t.Fatalf("error = %q, want %q", got, want)
		}
	})
}

func TestOrderCommitActionsPhasesDependencies(t *testing.T) {
	actions := []CommitAction{
		{TrackID: "merge-track", Action: "merge"},
		{TrackID: "variant-track", Action: "new_variant"},
		{TrackID: "identity-track", Action: "new_identity"},
	}

	ordered := orderCommitActions(actions)
	if len(ordered) != len(actions) {
		t.Fatalf("expected %d ordered actions, got %d", len(actions), len(ordered))
	}

	got := []string{ordered[0].TrackID, ordered[1].TrackID, ordered[2].TrackID}
	want := []string{"identity-track", "variant-track", "merge-track"}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("ordered track IDs = %v, want %v", got, want)
		}
	}
}

func TestOrderCommitActionsPreservesRelativeOrderWithinPhase(t *testing.T) {
	actions := []CommitAction{
		{TrackID: "identity-1", Action: "new_identity"},
		{TrackID: "merge-1", Action: "merge"},
		{TrackID: "identity-2", Action: "new_identity"},
		{TrackID: "variant-1", Action: "new_variant"},
		{TrackID: "merge-2", Action: "merge"},
	}

	ordered := orderCommitActions(actions)
	got := []string{ordered[0].TrackID, ordered[1].TrackID, ordered[2].TrackID, ordered[3].TrackID, ordered[4].TrackID}
	want := []string{"identity-1", "identity-2", "variant-1", "merge-1", "merge-2"}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("ordered track IDs = %v, want %v", got, want)
		}
	}
}

func TestIdentityNamePatternMatchesCaseInsensitiveSystemLabels(t *testing.T) {
	cases := []string{"Identity 42", "identity 42", "IDENTITY 42"}
	for _, tc := range cases {
		matches := identityNamePattern.FindStringSubmatch(tc)
		if len(matches) != 2 || matches[1] != "42" {
			t.Fatalf("expected %q to match system identity pattern, got %v", tc, matches)
		}
	}
}
