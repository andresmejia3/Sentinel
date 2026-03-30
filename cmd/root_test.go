package cmd

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadEnvFileSetsUnsetVars(t *testing.T) {
	dir := t.TempDir()
	envPath := filepath.Join(dir, ".env")
	content := "POSTGRES_USER=sentinel\nPOSTGRES_PASSWORD=\"secret\"\nexport POSTGRES_DB=sentinel\n"

	if err := os.WriteFile(envPath, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	t.Setenv("POSTGRES_USER", "")
	t.Setenv("POSTGRES_PASSWORD", "")
	t.Setenv("POSTGRES_DB", "")

	if err := os.Unsetenv("POSTGRES_USER"); err != nil {
		t.Fatal(err)
	}
	if err := os.Unsetenv("POSTGRES_PASSWORD"); err != nil {
		t.Fatal(err)
	}
	if err := os.Unsetenv("POSTGRES_DB"); err != nil {
		t.Fatal(err)
	}

	if err := loadLocalEnvFromPaths([]string{envPath}); err != nil {
		t.Fatalf("loadLocalEnvFromPaths returned error: %v", err)
	}

	if got := os.Getenv("POSTGRES_USER"); got != "sentinel" {
		t.Fatalf("POSTGRES_USER = %q, want %q", got, "sentinel")
	}
	if got := os.Getenv("POSTGRES_PASSWORD"); got != "secret" {
		t.Fatalf("POSTGRES_PASSWORD = %q, want %q", got, "secret")
	}
	if got := os.Getenv("POSTGRES_DB"); got != "sentinel" {
		t.Fatalf("POSTGRES_DB = %q, want %q", got, "sentinel")
	}
}

func TestLoadEnvFileDoesNotOverrideExistingVars(t *testing.T) {
	dir := t.TempDir()
	envPath := filepath.Join(dir, ".env")
	content := "POSTGRES_HOST=db\n"

	if err := os.WriteFile(envPath, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	t.Setenv("POSTGRES_HOST", "localhost")

	if err := loadLocalEnvFromPaths([]string{envPath}); err != nil {
		t.Fatalf("loadLocalEnvFromPaths returned error: %v", err)
	}

	if got := os.Getenv("POSTGRES_HOST"); got != "localhost" {
		t.Fatalf("POSTGRES_HOST = %q, want %q", got, "localhost")
	}
}

func TestLoadEnvFileRejectsMalformedLines(t *testing.T) {
	dir := t.TempDir()
	envPath := filepath.Join(dir, ".env")

	if err := os.WriteFile(envPath, []byte("THIS IS NOT VALID\n"), 0644); err != nil {
		t.Fatal(err)
	}

	if err := loadLocalEnvFromPaths([]string{envPath}); err == nil {
		t.Fatal("expected malformed .env to return an error")
	}
}
