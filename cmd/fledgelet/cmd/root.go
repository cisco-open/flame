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

		apiServerInfo := objects.ServerInfo{IP: restIp, Port: restPort}
		notifierInfo := objects.ServerInfo{IP: notifyIp, Port: notifyPort}
		agent, err := app.NewAgent(apiServerInfo, notifierInfo)
		if err != nil {
			return err
		}

		if err := agent.Start(); err != nil {
			return err
		}

		return nil
	},
}

func init() {
	agentCmd.Flags().StringP("notifyIp", "i", "0.0.0.0", "Notifier IP")
	agentCmd.Flags().Uint16P("notifyPort", "p", util.NotifierGrpcPort, "Notifier port")
	agentCmd.Flags().StringP("apiIp", "a", "0.0.0.0", "REST API IP")
	agentCmd.Flags().Uint16P("apiPort", "o", util.ApiServerRestApiPort, "REST API port")

	agentCmd.MarkFlagRequired("notifyIp")
	agentCmd.MarkFlagRequired("apiIp")
}

func Execute() error {
	return agentCmd.Execute()
}
