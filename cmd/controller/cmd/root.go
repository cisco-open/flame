package cmd

import (
	"github.com/spf13/cobra"

	"wwwin-github.cisco.com/eti/fledge/cmd/controller/app"
	"wwwin-github.cisco.com/eti/fledge/pkg/openapi"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

var rootCmd = &cobra.Command{
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

		notificationInfo := openapi.ServerInfo{Ip: notifyIp, Port: int32(notifyPort)}
		if err := app.StartController(uri, notificationInfo); err != nil {
			return err
		}
		return nil
	},
}

func init() {
	rootCmd.PersistentFlags().StringP("notifyIp", "i", "0.0.0.0", "Notifer IP")
	rootCmd.PersistentFlags().Uint16P("notifyPort", "p", util.NotifierGrpcPort, "Notifier port")
	rootCmd.PersistentFlags().StringP("uri", "u", "", "Database connection URI")
	rootCmd.MarkPersistentFlagRequired("uri")
}

func Execute() error {
	return rootCmd.Execute()
}
