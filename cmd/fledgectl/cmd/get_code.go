package cmd

import (
	"github.com/spf13/cobra"

	"wwwin-github.cisco.com/eti/fledge/cmd/fledgectl/resources/code"
)

var getDesignCodeCmd = &cobra.Command{
	Use:   "code <version>",
	Short: "Get an ML code",
	Long:  "This command retrieves an ML code for a design",
	Args:  cobra.RangeArgs(1, 1),
	RunE: func(cmd *cobra.Command, args []string) error {
		codeVer := args[0]

		flags := cmd.Flags()

		designId, err := flags.GetString("design")
		if err != nil {
			return err
		}

		params := code.Params{}
		params.Host = config.ApiServer.Host
		params.Port = config.ApiServer.Port
		params.User = config.User
		params.DesignId = designId
		params.CodeVer = codeVer

		return code.Get(params)
	},
}

func init() {
	getDesignCodeCmd.PersistentFlags().StringP("design", "d", "", "Design ID")
	getDesignCodeCmd.MarkPersistentFlagRequired("design")
	getCmd.AddCommand(getDesignCodeCmd)
}
