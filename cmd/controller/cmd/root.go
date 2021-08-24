package cmd

import (
	"github.com/spf13/cobra"

	"wwwin-github.cisco.com/eti/fledge/cmd/controller/app"
	"wwwin-github.cisco.com/eti/fledge/cmd/controller/database"
	"wwwin-github.cisco.com/eti/fledge/pkg/objects"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

var ctlrServerCmd = &cobra.Command{
	Use:   "ctlrServer",
	Short: "Controller server",
	Run: func(cmd *cobra.Command, args []string) {
		cmd.Help()
	},
}

var startControllerServerCmd = &cobra.Command{
	Use:   "start",
	Short: "Start API server",
	Long:  "Start API server",
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

		db, err := flags.GetString("db")
		if err != nil {
			return err
		}

		uri, err := flags.GetString("uri")
		if err != nil {
			return err
		}

		dbInfo := database.DBInfo{Name: db, URI: uri}
		notificationInfo := objects.ServerInfo{IP: notifyIp, Port: notifyPort}
		if err := app.StartController(dbInfo, notificationInfo); err != nil {
			return err
		}
		return nil
	},
}

func init() {
	ctlrServerCmd.AddCommand(startControllerServerCmd)

	ctlrServerCmd.PersistentFlags().StringP("notifyIp", "i", "0.0.0.0", "Notifer IP")
	ctlrServerCmd.PersistentFlags().Uint16P("notifyPort", "p", util.NotifierGrpcPort, "Notifier port")
	ctlrServerCmd.PersistentFlags().StringP("db", "d", util.MONGODB, "Database type")
	ctlrServerCmd.PersistentFlags().StringP("uri", "u", "", "Database connection URI")
	ctlrServerCmd.MarkPersistentFlagRequired("uri")
}

func Execute() error {
	return ctlrServerCmd.Execute()
}
