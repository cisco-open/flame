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

	"github.com/cisco-open/flame/cmd/flamectl/resources/design"
)

var getDesignCmd = &cobra.Command{
	Use:   "design <designId>",
	Short: "Get the design template of a given design ID",
	Long:  "This command retrieves the design template of a given design ID",
	Args:  cobra.RangeArgs(1, 1),
	RunE: func(cmd *cobra.Command, args []string) error {
		designId := args[0]

		params := design.Params{}
		params.Endpoint = config.ApiServer.Endpoint
		params.User = config.User
		params.DesignId = designId

		return design.Get(params)
	},
}

var getDesignsCmd = &cobra.Command{
	Use:   "designs",
	Short: "Get a list of all design templates",
	Args:  cobra.RangeArgs(0, 0),
	Long:  "This command retrieves a list of all design templates",
	RunE: func(cmd *cobra.Command, args []string) error {
		flags := cmd.Flags()

		limit, err := flags.GetString("limit")
		if err != nil {
			return err
		}

		params := design.Params{}
		params.Endpoint = config.ApiServer.Endpoint
		params.User = config.User
		params.Limit = limit

		return design.GetMany(params)
	},
}

func init() {
	getDesignsCmd.Flags().StringP("limit", "l", "100", "List of all the designs by this user")

	getCmd.AddCommand(getDesignCmd, getDesignsCmd)
}
