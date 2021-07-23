package cmd

import (
	"github.com/spf13/cobra"

	"wwwin-github.cisco.com/eti/fledge/cmd/agent/app"
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
	Short: "Start agent",
	Long:  "Start agent",
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

		notificationInfo := objects.ServerInfo{IP: notifyIp, Port: notifyPort}

		if err := app.StartAgent(notificationInfo); err != nil {
			return err
		}
		return nil
	},
}

func init() {
	agentCmd.AddCommand(startAgentCmd)

	startAgentCmd.Flags().StringP("notifyIp", "i", "0.0.0.0", "Notification service ip")
	startAgentCmd.Flags().Uint16P("notifyPort", "p", util.NotificationServiceGrpcPort, "Notification service port")
	//TODO remove me after discussion - will obtain these details from environment variable?
	//startAgentCmd.Flags().StringP("name", "n", "agent", "Agent name")
	//startAgentCmd.Flags().StringP("uuid", "u", "0", "Agent uuid")

	startAgentCmd.MarkFlagRequired("notifyIp")
	//startAgentCmd.MarkFlagRequired("name")
	//startAgentCmd.MarkFlagRequired("uuid")
}

func Execute() error {
	return agentCmd.Execute()
}
