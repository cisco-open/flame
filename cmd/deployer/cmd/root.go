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

	"github.com/cisco-open/flame/cmd/deployer/app"
	"github.com/cisco-open/flame/pkg/openapi"
	"github.com/cisco-open/flame/pkg/util"
)

const (
	argApiserver = "apiserver"

	optionInsecure = "insecure"
	optionPlain    = "plain"
)

var rootCmd = &cobra.Command{
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

		bInsecure, _ := flags.GetBool(optionInsecure)
		bPlain, _ := flags.GetBool(optionPlain)

		if bInsecure && bPlain {
			err = fmt.Errorf("options --%s and --%s are incompatible; enable one of them", optionInsecure, optionPlain)
			return err
		}

		// Pass more values to NewCompute using
		computeSpec := openapi.ComputeSpec{
			AdminId:   "admin1",
			Region:    "us-east",
			ComputeId: "1",
			ApiKey:    "temp",
		}

		compute, err := app.NewCompute(apiserver, computeSpec, bInsecure, bPlain)
		if err != nil {
			return err
		}

		if err := compute.RegisterNewCompute(); err != nil {
			err = fmt.Errorf("error from RegisterNewCompute: %s", err)
			return err
		}

		select {}
	},
}

func init() {
	defaultApiServerEp := fmt.Sprintf("http://0.0.0.0:%d", util.ApiServerRestApiPort)
	rootCmd.Flags().StringP(argApiserver, "a", defaultApiServerEp, "API server endpoint")
	rootCmd.MarkFlagRequired(argApiserver)

	rootCmd.PersistentFlags().Bool(optionInsecure, false, "Allow insecure connection")
	rootCmd.PersistentFlags().Bool(optionPlain, false, "Allow unencrypted connection")
}

func Execute() error {
	return rootCmd.Execute()
}
