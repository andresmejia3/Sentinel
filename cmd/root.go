package cmd

import (
	"fmt"
	"os"
	"github.com/spf13/cobra"
)

var rootCmd = &cobra.Command{
	Use:   "sentinel",
	Short: `Sentinel is a high-speed forensic engine designed for Zero-Disk I/O 
			video processing. It utilizes a parallelized Python inference pool to 
			index biometric vectors and automate identity-based redaction with a 
			fail-safe privacy net.`,
}

func Execute() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}