package cmd

import (
	"github.com/spf13/cobra"
	"go.uber.org/zap"

	"wwwin-github.cisco.com/eti/fledge/cmd/apiserver/app"
	"wwwin-github.cisco.com/eti/fledge/pkg/openapi"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

/*
 * APPNAME COMMAND ARG --FLAG
 * example hugo server --port=1313 -- 'server' is a command, and 'port' is a flag
 * example codebase https://github.com/schadokar/my-calc
 */
var apiServerCmd = &cobra.Command{
	Use:   util.ApiServer,
	Short: "API server",
	Long:  "API server",
	Run: func(cmd *cobra.Command, args []string) {
		cmd.Help()
	},
}

var startApiServerCmd = &cobra.Command{
	Use:   "start",
	Short: "Start API server",
	Long:  "Start API server",
	RunE: func(cmd *cobra.Command, args []string) error {
		flags := cmd.Flags()
		portNo, err := flags.GetUint16("port")
		if err != nil {
			return err
		}

		ctlrIp, err := flags.GetString("ctlrIp")
		if err != nil {
			return err
		}

		ctlrPort, err := flags.GetUint16("ctlrPort")
		if err != nil {
			return err
		}

		if err := app.RunServer(portNo, openapi.ServerInfo{
			Ip:   ctlrIp,
			Port: int32(ctlrPort),
		}); err != nil {
			return err
		}
		return nil
	},
}

var stopApiServerCmd = &cobra.Command{
	Use:   "stop",
	Short: "Stop API server",
	Long:  "Stop API server",
	Run: func(cmd *cobra.Command, args []string) {
		zap.S().Infof("Stopping API server ...")
	},
}

var reloadApiServerCmd = &cobra.Command{
	Use:   "reload",
	Short: "Reload API server",
	Long:  "Reload API server",
	Run: func(cmd *cobra.Command, args []string) {
		zap.S().Infof("Reloading API server ...")
	},
}

/**
 * This is the first function which gets called whenever a package initialize in the golang.
 * https://towardsdatascience.com/how-to-create-a-cli-in-golang-with-cobra-d729641c7177
 * The order of function calls is as follow
 * 	init --> main --> cobra.OnInitialize --> command
 */
func init() {
	//todo use systemd and make the server daemon process.
	apiServerCmd.AddCommand(startApiServerCmd, stopApiServerCmd, reloadApiServerCmd)

	apiServerCmd.PersistentFlags().Uint16P("port", "p", util.ApiServerRestApiPort, "listening port for API server")

	startApiServerCmd.Flags().StringP("ctlrIp", "c", "0.0.0.0", "Controller ip")
	startApiServerCmd.Flags().Uint16P("ctlrPort", "o", util.ControllerRestApiPort, "Controller port")
	startApiServerCmd.MarkFlagRequired("ctlrIp")
}

func Execute() error {
	return apiServerCmd.Execute()
}
