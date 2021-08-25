package cmd

import (
	"github.com/spf13/cobra"

	"wwwin-github.cisco.com/eti/fledge/cmd/controller/app"
	"wwwin-github.cisco.com/eti/fledge/pkg/objects"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

var ctlrServerCmd = &cobra.Command{
	Use:   util.Controller,
	Short: util.ProjectName + util.Controller,
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

		uri, err := flags.GetString("uri")
		if err != nil {
			return err
		}

		notificationInfo := objects.ServerInfo{IP: notifyIp, Port: notifyPort}
		if err := app.StartController(uri, notificationInfo); err != nil {
			return err
		}
		return nil
	},
}

func init() {
	ctlrServerCmd.PersistentFlags().StringP("notifyIp", "i", "0.0.0.0", "Notifer IP")
	ctlrServerCmd.PersistentFlags().Uint16P("notifyPort", "p", util.NotifierGrpcPort, "Notifier port")
	ctlrServerCmd.PersistentFlags().StringP("uri", "u", "", "Database connection URI")
	ctlrServerCmd.MarkPersistentFlagRequired("uri")
}

func Execute() error {
	return ctlrServerCmd.Execute()
}
