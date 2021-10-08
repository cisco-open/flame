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

	"github.com/cisco/fledge/cmd/controller/app"
	"github.com/cisco/fledge/cmd/controller/app/deployer"
	"github.com/cisco/fledge/pkg/util"
)

var rootCmd = &cobra.Command{
	Use:   util.Controller,
	Short: util.ProjectName + util.Controller,
	RunE: func(cmd *cobra.Command, args []string) error {
		flags := cmd.Flags()

		dbUri, err := flags.GetString("db")
		if err != nil {
			return err
		}

		notifier, err := flags.GetString("notifier")
		if err != nil {
			return err
		}

		port, err := flags.GetString("port")
		if err != nil {
			return err
		}

		platform, err := flags.GetString("platform")
		if err != nil {
			return err
		}

		instance, err := app.NewController(dbUri, notifier, port, platform)
		if err != nil {
			fmt.Printf("Failed to create a new controller instance\n")
			return nil
		}
		instance.Start()

		return nil
	},
}

func init() {
	rootCmd.PersistentFlags().StringP("db", "d", "", "Database connection URI")
	rootCmd.MarkPersistentFlagRequired("db")

	defaultEndpoint := fmt.Sprintf("0.0.0.0:%d", util.NotifierGrpcPort)
	rootCmd.PersistentFlags().StringP("notifier", "n", defaultEndpoint, "Notifier endpoint")

	defaultCtrlPort := fmt.Sprintf("%d", util.ControllerRestApiPort)
	rootCmd.PersistentFlags().StringP("port", "p", defaultCtrlPort, "Port for controller's REST API service")

	desc := fmt.Sprintf("deployment platform; options: [ %s | %s ]", deployer.K8S, deployer.DOCKER)
	rootCmd.PersistentFlags().StringP("platform", "", deployer.K8S, desc)
}

func Execute() error {
	return rootCmd.Execute()
}
