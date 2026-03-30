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
