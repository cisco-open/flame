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
	"github.com/spf13/cobra"

	"github.com/cisco-open/flame/cmd/notifier/app"
	"github.com/cisco-open/flame/pkg/util"
)

var rootCmd = &cobra.Command{
	Use:   util.Notifier,
	Short: util.ProjectName + " " + util.Notifier,
	RunE: func(cmd *cobra.Command, args []string) error {
		flags := cmd.Flags()

		port, err := flags.GetUint16("port")
		if err != nil {
			return err
		}

		//start GRPC service
		app.StartGRPCService(port)

		return nil
	},
}

func init() {
	rootCmd.Flags().Uint16P("port", "p", util.NotifierGrpcPort, "service port")
}

func Execute() error {
	return rootCmd.Execute()
}
