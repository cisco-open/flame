// Copyright 2024 Cisco Systems, Inc. and its affiliates
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

var updateDesignCmd = &cobra.Command{
	Use:   "design <designId> [--description <desc>] [--name <name>]",
	Short: "Update a design",
	Long:  "This command updates a design",
	Args:  cobra.RangeArgs(1, 1),
	RunE: func(cmd *cobra.Command, args []string) error {
		checkInsecure(cmd)

		designId := args[0]

		flags := cmd.Flags()
		description, descErr := flags.GetString("desc")
		name, nameErr := flags.GetString("name")

		if descErr != nil {
			return descErr
		}

		if nameErr != nil {
			return nameErr
		}

		params := design.Params{}
		params.Endpoint = config.ApiServer.Endpoint
		params.User = config.User
		params.DesignId = designId
		params.Desc = description
		params.Name = name

		return design.Update(params)
	},
}

func init() {
	updateDesignCmd.Flags().StringP("name", "n", "", "Design name")
	updateDesignCmd.Flags().StringP("desc", "d", "", "Design description")
	updateCmd.AddCommand(updateDesignCmd)
}
