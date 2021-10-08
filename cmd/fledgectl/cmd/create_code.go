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
	"github.com/spf13/cobra"

	"github.com/cisco/fledge/cmd/fledgectl/resources/code"
)

var createDesignCodeCmd = &cobra.Command{
	Use:   "code",
	Short: "Create a new ML code for a design",
	Long:  "Command to create a new ML code for a design",
	Args:  cobra.RangeArgs(0, 0),
	RunE: func(cmd *cobra.Command, args []string) error {
		flags := cmd.Flags()

		designId, err := flags.GetString("design")
		if err != nil {
			return err
		}

		codePath, err := flags.GetString("path")
		if err != nil {
			return err
		}

		params := code.Params{}
		params.Endpoint = config.ApiServer.Endpoint
		params.User = config.User
		params.DesignId = designId
		params.CodePath = codePath
		params.CodeVer = ""

		return code.Create(params)
	},
}

func init() {
	createDesignCodeCmd.PersistentFlags().StringP("design", "d", "", "Design ID")
	createDesignCodeCmd.MarkPersistentFlagRequired("design")
	createDesignCodeCmd.PersistentFlags().StringP("path", "p", "", "Path to a zipped ML code file")
	createDesignCodeCmd.MarkPersistentFlagRequired("path")
	createCmd.AddCommand(createDesignCodeCmd)
}
