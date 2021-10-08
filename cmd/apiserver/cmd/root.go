// Copyright (c) 2021 Cisco Systems, Inc. and its affiliates
// All rights reserved
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package cmd

import (
	"fmt"

	"github.com/spf13/cobra"

	"github.com/cisco/fledge/cmd/apiserver/app"
	"github.com/cisco/fledge/pkg/util"
)

/*
 * APPNAME COMMAND ARG --FLAG
 * example hugo server --port=1313 -- 'server' is a command, and 'port' is a flag
 * example codebase https://github.com/schadokar/my-calc
 */
var rootCmd = &cobra.Command{
	Use:   util.ApiServer,
	Short: "API server",
	Long:  "API server",
	RunE: func(cmd *cobra.Command, args []string) error {
		flags := cmd.Flags()
		portNo, err := flags.GetUint16("port")
		if err != nil {
			return err
		}

		controller, err := flags.GetString("controller")
		if err != nil {
			return err
		}

		_ = app.RunServer(portNo, controller)

		return nil
	},
}

/**
 * This is the first function which gets called whenever a package initialize in the golang.
 * https://towardsdatascience.com/how-to-create-a-cli-in-golang-with-cobra-d729641c7177
 * The order of function calls is as follow
 * 	init --> main --> cobra.OnInitialize --> command
 */
func init() {
	rootCmd.PersistentFlags().Uint16P("port", "p", util.ApiServerRestApiPort, "listening port for API server")

	defaultControllerEp := fmt.Sprintf("0.0.0.0:%d", util.ControllerRestApiPort)
	rootCmd.Flags().StringP("controller", "c", defaultControllerEp, "Controller endpoint")
	rootCmd.MarkFlagRequired("controller")
}

func Execute() error {
	return rootCmd.Execute()
}
