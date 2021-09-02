package cmd

import "github.com/spf13/cobra"

var createCmd = &cobra.Command{
	Use:   "create <resource> <name>",
	Short: "Command to create resources",
	Run: func(cmd *cobra.Command, args []string) {
		cmd.Help()
	},
}

func init() {
	rootCmd.AddCommand(createCmd)
}
