package cmd

import (
	"github.com/spf13/cobra"

	"wwwin-github.cisco.com/eti/fledge/cmd/fledgectl/resources/design"
)

var createDesignCmd = &cobra.Command{
	Use:   "design <designId>",
	Short: "Create a new design template",
	Long:  "This command creates a new design template",
	Args:  cobra.RangeArgs(1, 1),
	RunE: func(cmd *cobra.Command, args []string) error {
		designId := args[0]

		flags := cmd.Flags()
		description, err := flags.GetString("desc")
		if err != nil {
			return err
		}

		params := design.Params{}
		params.Host = config.ApiServer.Host
		params.Port = config.ApiServer.Port
		params.User = config.User
		params.DesignId = designId
		params.Desc = description

		return design.Create(params)
	},
}

func init() {
	createDesignCmd.Flags().StringP("desc", "d", "", "Design description")
	createCmd.AddCommand(createDesignCmd)
}
