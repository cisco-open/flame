// Copyright 2022 Cisco Systems, Inc. and its affiliates
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

package cmd

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/spf13/cobra"

	"github.com/cisco-open/flame/cmd/deployer/app"
	"github.com/cisco-open/flame/pkg/openapi"
	"github.com/cisco-open/flame/pkg/util"
)

const (
	argApiserver   = "apiserver"
	argNotifier    = "notifier"
	argConf        = "conf"
	optionInsecure = "insecure"
	optionPlain    = "plain"
)

type DeployerRequest struct {
	ComputeSpec openapi.ComputeSpec `json:"computeSpec"`
	Platform    string              `json:"platform"`
	Namespace   string              `json:"namespace"`
}

var (
	confFile string

	rootCmd = &cobra.Command{
		Use:   util.Deployer,
		Short: util.ProjectName + " Deployer",
		RunE: func(cmd *cobra.Command, args []string) error {
			flags := cmd.Flags()

			apiserver, err := flags.GetString(argApiserver)
			if err != nil {
				return err
			}
			if len(strings.Split(apiserver, ":")) != util.NumTokensInRestEndpoint {
				return fmt.Errorf("incorrect format for apiserver endpoint: %s", apiserver)
			}

			notifier, err := flags.GetString(argNotifier)
			if err != nil {
				return err
			}
			if len(strings.Split(notifier, ":")) != util.NumTokensInEndpoint {
				return fmt.Errorf("incorrect format for notifier endpoint: %s", notifier)
			}

			data, err := os.ReadFile(confFile)
			if err != nil {
				fmt.Printf("Failed to read deployer configuration file %s: %v\n", confFile, err)
				return nil
			}
			fmt.Printf("deployer configuration %v\n", string(data))

			// encode the data
			deployerRequest := DeployerRequest{}
			err = json.Unmarshal(data, &deployerRequest)
			if err != nil {
				fmt.Println("Failed to unmarshal the deployer configuration file")
				return nil
			}

			bInsecure, _ := flags.GetBool(optionInsecure)
			bPlain, _ := flags.GetBool(optionPlain)

			if bInsecure && bPlain {
				err = fmt.Errorf("options --%s and --%s are incompatible; enable one of them", optionInsecure, optionPlain)
				return err
			}

			compute, err := app.NewCompute(apiserver, deployerRequest.ComputeSpec, bInsecure, bPlain)
			if err != nil {
				return err
			}

			err = compute.RegisterNewCompute()
			if err != nil {
				err = fmt.Errorf("unable to register new compute with controller: %s", err)
				return err
			}

			resourceHandler := app.NewResourceHandler(apiserver, notifier, deployerRequest.ComputeSpec, deployerRequest.Platform, deployerRequest.Namespace, bInsecure, bPlain)
			resourceHandler.Start()

			select {}
		},
	}
)

func init() {
	defaultApiServerEp := fmt.Sprintf("http://0.0.0.0:%d", util.ApiServerRestApiPort)
	rootCmd.Flags().StringP(argApiserver, "a", defaultApiServerEp, "API server endpoint")
	rootCmd.MarkFlagRequired(argApiserver)

	defaultNotifierEp := fmt.Sprintf("0.0.0.0:%d", util.NotifierGrpcPort)
	rootCmd.Flags().StringP(argNotifier, "n", defaultNotifierEp, "Notifier endpoint")
	rootCmd.MarkFlagRequired(argNotifier)

	rootCmd.Flags().StringVar(&confFile, "config", "", "Deployer configuration file")
	rootCmd.MarkFlagRequired(argConf)

	rootCmd.PersistentFlags().Bool(optionInsecure, false, "Allow insecure connection")
	rootCmd.PersistentFlags().Bool(optionPlain, false, "Allow unencrypted connection")
}

func Execute() error {
	return rootCmd.Execute()
}
