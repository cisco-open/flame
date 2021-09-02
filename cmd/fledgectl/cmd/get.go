package cmd

import "github.com/spf13/cobra"

var getCmd = &cobra.Command{
	Use:   "get <resource> <name>",
	Short: "Command to get resources",
	Run: func(cmd *cobra.Command, args []string) {
		cmd.Help()
	},
}

func init() {
	rootCmd.AddCommand(getCmd)
}
