package cmd

import (
	"github.com/spf13/cobra"

	"wwwin-github.cisco.com/eti/fledge/cmd/fledgelet/app"
	"wwwin-github.cisco.com/eti/fledge/pkg/objects"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

var agentCmd = &cobra.Command{
	Use:   util.Agent,
	Short: util.ProjectName + " Agent",
	Run: func(cmd *cobra.Command, args []string) {
		cmd.Help()
	},
}

var startAgentCmd = &cobra.Command{
	Use:   "start",
	Short: "Start fledgelet",
	Long:  "Start fledgelet",
	RunE: func(cmd *cobra.Command, args []string) error {
		flags := cmd.Flags()

		notifyIp, err := flags.GetString("notifyIp")
		if err != nil {
			return err
		}

		notifyPort, err := flags.GetUint16("notifyPort")
		if err != nil {
			return err
		}

		restIp, err := flags.GetString("apiIp")
		if err != nil {
			return err
		}

		restPort, err := flags.GetUint16("apiPort")
		if err != nil {
			return err
		}

		if err := app.StartAgent(objects.ServerInfo{IP: notifyIp, Port: notifyPort}, objects.ServerInfo{IP: restIp, Port: restPort}); err != nil {
			return err
		}
		return nil
	},
}

func init() {
	agentCmd.AddCommand(startAgentCmd)

	startAgentCmd.Flags().StringP("notifyIp", "i", "0.0.0.0", "Notification service ip")
	startAgentCmd.Flags().Uint16P("notifyPort", "p", util.NotificationServiceGrpcPort, "Notification service port")
	startAgentCmd.Flags().StringP("apiIp", "a", "0.0.0.0", "REST API ip")
	startAgentCmd.Flags().Uint16P("apiPort", "o", util.ApiServerRestApiPort, "REST API port")

	startAgentCmd.MarkFlagRequired("notifyIp")
	startAgentCmd.MarkFlagRequired("apiIp")
}

func Execute() error {
	return agentCmd.Execute()
}
