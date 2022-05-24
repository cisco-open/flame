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
	"fmt"
	"strings"

	"github.com/spf13/cobra"

	"github.com/cisco-open/flame/cmd/flamelet/app"
	"github.com/cisco-open/flame/pkg/util"
)

const (
	argApiserver = "apiserver"
	argNotifier  = "notifier"

	optionInsecure = "insecure"
	optionPlain    = "plain"
)

var rootCmd = &cobra.Command{
	Use:   util.Agent,
	Short: util.ProjectName + " Agent",
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

		bInsecure, _ := flags.GetBool(optionInsecure)
		bPlain, _ := flags.GetBool(optionPlain)

		if bInsecure && bPlain {
			err = fmt.Errorf("options --%s and --%s are incompatible; enable one of them.\n", optionInsecure, optionPlain)
			return err
		}

		agent, err := app.NewAgent(apiserver, notifier, bInsecure, bPlain)
		if err != nil {
			return err
		}

		if err := agent.Start(); err != nil {
			return err
		}

		return nil
	},
}

func init() {
	defaultApiServerEp := fmt.Sprintf("http://0.0.0.0:%d", util.ApiServerRestApiPort)
	rootCmd.Flags().StringP(argApiserver, "a", defaultApiServerEp, "API server endpoint")
	rootCmd.MarkFlagRequired(argApiserver)

	defaultNotifierEp := fmt.Sprintf("0.0.0.0:%d", util.NotifierGrpcPort)
	rootCmd.Flags().StringP(argNotifier, "n", defaultNotifierEp, "Notifier endpoint")
	rootCmd.MarkFlagRequired(argNotifier)

	rootCmd.PersistentFlags().Bool(optionInsecure, false, "Allow insecure connection")
	rootCmd.PersistentFlags().Bool(optionPlain, false, "Allow unencrypted connection")
}

func Execute() error {
	return rootCmd.Execute()
}
