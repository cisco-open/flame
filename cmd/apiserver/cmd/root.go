package cmd

import (
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
	klog "k8s.io/klog/v2"

	"wwwin-github.cisco.com/fledge/fledge/cmd/apiserver/app"
)

var rootCmd = &cobra.Command{
	Use:   "fledge-apiserver",
	Short: "fledge REST API server",
	RunE: func(cmd *cobra.Command, args []string) error {
		flags := cmd.Flags()

		flags.VisitAll(func(flag *pflag.Flag) {
			klog.Infof("FLAG: --%s=%q", flag.Name, flag.Value)
		})

		portNo, err := flags.GetUint16("port")
		if err != nil {
			return err
		}

		if err := app.RunServer(portNo); err != nil {
			return err
		}
		return nil
	},
}

func init() {
	rootCmd.PersistentFlags().Uint16P("port", "p", 63575, "listening port for fledge-apiserver")
}

func Execute() error {
	return rootCmd.Execute()
}
