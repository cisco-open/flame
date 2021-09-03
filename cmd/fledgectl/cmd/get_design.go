package cmd

import (
	"github.com/spf13/cobra"

	"wwwin-github.cisco.com/eti/fledge/cmd/fledgectl/resources/design"
)

var getDesignCmd = &cobra.Command{
	Use:   "design <designId>",
	Short: "Get the design template of a given design ID",
	Long:  "This command retrieves the design template of a given design ID",
	Args:  cobra.RangeArgs(1, 1),
	RunE: func(cmd *cobra.Command, args []string) error {
		designId := args[0]

		params := design.Params{
			Host:     config.ApiServer.Host,
			Port:     config.ApiServer.Port,
			User:     config.User,
			DesignId: designId,
		}

		return design.Get(params)
	},
}

var getDesignsCmd = &cobra.Command{
	Use:   "designs",
	Short: "Get a list of all design templates",
	Args:  cobra.RangeArgs(0, 0),
	Long:  "This command retrieves a list of all design templates",
	RunE: func(cmd *cobra.Command, args []string) error {
		flags := cmd.Flags()

		limit, err := flags.GetString("limit")
		if err != nil {
			return err
		}

		params := design.Params{
			Host:  config.ApiServer.Host,
			Port:  config.ApiServer.Port,
			User:  config.User,
			Limit: limit,
		}

		return design.GetMany(params)
	},
}

func init() {
	getDesignsCmd.Flags().StringP("limit", "l", "100", "List of all the designs by this user")

	getCmd.AddCommand(getDesignCmd, getDesignsCmd)
}
