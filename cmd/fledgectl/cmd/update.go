package cmd

import "github.com/spf13/cobra"

var updateCmd = &cobra.Command{
	Use:   "update <resource> <name>",
	Short: "Command to update resources",
	Run: func(cmd *cobra.Command, args []string) {
		cmd.Help()
	},
}

func init() {
	rootCmd.AddCommand(updateCmd)
}
