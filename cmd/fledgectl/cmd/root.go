package cmd

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"
	"go.uber.org/zap"

	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

/*
 * APPNAME COMMAND ARG --FLAG
 * example hugo server --port=1313 -- 'server' is a command, and 'port' is a flag
 * example codebase https://github.com/schadokar/my-calc
 */
var (
	cfgFile string
	config  *Config

	rootCmd = &cobra.Command{
		Use:   util.CliTool,
		Short: util.ProjectName + " CLI Tool",
		Run: func(cmd *cobra.Command, args []string) {
			cmd.Help()
		},
	}
)

/**
 * This is the first function which gets called whenever a package initialize in the golang.
 * https://towardsdatascience.com/how-to-create-a-cli-in-golang-with-cobra-d729641c7177
 * The order of function calls is as follow
 * 	init --> main --> cobra.OnInitialize --> command
 */
func init() {
	cobra.OnInitialize(initConfig)

	rootCmd.PersistentFlags().StringVar(&cfgFile, "config", "", "config file (default: $HOME/.fledge/config.yaml)")
}

func initConfig() {
	if cfgFile == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			zap.S().Fatalf("Failed to obtain home directory: %v", err)
		}
		cfgFile = filepath.Join(home, ".fledge", "config.yaml")
	}

	var err error

	config, err = loadConfig(cfgFile)
	if err != nil {
		zap.S().Fatalf("Failed to load config %s: %v", cfgFile, err)
	}
}

func Execute() error {
	return rootCmd.Execute()
}

func printCmdInfo(ip string, portNo uint16, url string) {
	separator := "- - - - - - - - - - - - - - -"
	fmt.Printf("%s\nServer: %s:%d\n", separator, ip, portNo)
	fmt.Printf("URL: %s\n%s\n", url, separator)
}
