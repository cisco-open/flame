package cmd

import (
	"github.com/spf13/cobra"
	"go.uber.org/zap"
	"wwwin-github.cisco.com/fledge/fledge/cmd/apiserver/app"
	"wwwin-github.cisco.com/fledge/fledge/cmd/controller/database"
	util2 "wwwin-github.cisco.com/fledge/fledge/pkg/util"
)

/*
 * APPNAME COMMAND ARG --FLAG
 * example hugo server --port=1313 -- 'server' is a command, and 'port' is a flag
 * example codebase https://github.com/schadokar/my-calc
 */
var apiServerCmd = &cobra.Command{
	Use:   "apiServer",
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
		//flags.VisitAll(func(flag *pflag.Flag) {
		//	zap.S().Infof("FLAG: --%s=%q", flag.Name, flag.Value)
		//})

		portNo, err := flags.GetUint16("port")
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

		if err := app.RunServer(portNo, database.DBInfo{Name: db, URI: uri}); err != nil {
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

	apiServerCmd.PersistentFlags().Uint16P("port", "p", 5000, "listening port for API server")
	apiServerCmd.PersistentFlags().StringP("db", "d", util2.MONGODB, "Database type")
	apiServerCmd.PersistentFlags().StringP("uri", "u", "", "Database connection URI")
	apiServerCmd.MarkPersistentFlagRequired("uri")
}

func Execute() error {
	return apiServerCmd.Execute()
}
