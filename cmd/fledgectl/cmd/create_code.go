package cmd

import (
	"github.com/spf13/cobra"

	"wwwin-github.cisco.com/eti/fledge/cmd/fledgectl/resources/code"
)

var createDesignCodeCmd = &cobra.Command{
	Use:   "code",
	Short: "Create a new ML code for a design",
	Long:  "Command to create a new ML code for a design",
	Args:  cobra.RangeArgs(0, 0),
	RunE: func(cmd *cobra.Command, args []string) error {
		flags := cmd.Flags()

		designId, err := flags.GetString("design")
		if err != nil {
			return err
		}

		codePath, err := flags.GetString("path")
		if err != nil {
			return err
		}

		params := code.Params{}
		params.Host = config.ApiServer.Host
		params.Port = config.ApiServer.Port
		params.User = config.User
		params.DesignId = designId
		params.CodePath = codePath
		params.CodeVer = ""

		return code.Create(params)
	},
}

func init() {
	createDesignCodeCmd.PersistentFlags().StringP("design", "d", "", "Design ID")
	createDesignCodeCmd.MarkPersistentFlagRequired("design")
	createDesignCodeCmd.PersistentFlags().StringP("path", "p", "", "Path to a zipped ML code file")
	createDesignCodeCmd.MarkPersistentFlagRequired("path")
	createCmd.AddCommand(createDesignCodeCmd)
}
