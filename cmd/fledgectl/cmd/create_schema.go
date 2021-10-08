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

	"github.com/cisco/fledge/cmd/fledgectl/resources/schema"
)

var createDesignSchemaCmd = &cobra.Command{
	Use:   "schema",
	Short: "Create a new design schema",
	Long:  "Command to create a new design schema",
	Args:  cobra.RangeArgs(0, 0),
	RunE: func(cmd *cobra.Command, args []string) error {
		flags := cmd.Flags()

		designId, err := flags.GetString("design")
		if err != nil {
			return err
		}

		schemaPath, err := flags.GetString("path")
		if err != nil {
			return err
		}

		params := schema.Params{}
		params.Endpoint = config.ApiServer.Endpoint
		params.User = config.User
		params.DesignId = designId
		params.SchemaPath = schemaPath

		return schema.Create(params)
	},
}

func init() {
	createDesignSchemaCmd.PersistentFlags().StringP("design", "d", "", "Design ID")
	createDesignSchemaCmd.MarkPersistentFlagRequired("design")
	createDesignSchemaCmd.PersistentFlags().StringP("path", "p", "", "Path to schema json file")
	createDesignSchemaCmd.MarkPersistentFlagRequired("path")
	createCmd.AddCommand(createDesignSchemaCmd)
}
