package cmd

import (
	"github.com/spf13/cobra"

	"wwwin-github.cisco.com/eti/fledge/cmd/fledgectl/resources/dataset"
)

var createDatasetCmd = &cobra.Command{
	Use:   "dataset <dataset json file>",
	Short: "Create a new dataset",
	Long:  "This command creates a new dataset",
	Args:  cobra.RangeArgs(1, 1),
	RunE: func(cmd *cobra.Command, args []string) error {
		datasetFile := args[0]

		params := dataset.Params{}
		params.Host = config.ApiServer.Host
		params.Port = config.ApiServer.Port
		params.User = config.User
		params.DatasetFile = datasetFile

		return dataset.Create(params)
	},
}

func init() {
	createCmd.AddCommand(createDatasetCmd)
}
