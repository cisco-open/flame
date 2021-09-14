package cmd

import (
	"github.com/spf13/cobra"

	"wwwin-github.cisco.com/eti/fledge/cmd/fledgectl/resources/job"
)

var createJobCmd = &cobra.Command{
	Use:   "job <job json file>",
	Short: "Create a new job",
	Long:  "This command creates a new job",
	Args:  cobra.RangeArgs(1, 1),
	RunE: func(cmd *cobra.Command, args []string) error {
		jobFile := args[0]

		params := job.Params{}
		params.Host = config.ApiServer.Host
		params.Port = config.ApiServer.Port
		params.User = config.User
		params.JobFile = jobFile

		return job.Create(params)
	},
}

func init() {
	createCmd.AddCommand(createJobCmd)
}
