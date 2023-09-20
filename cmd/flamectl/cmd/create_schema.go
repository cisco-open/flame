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

	"github.com/cisco-open/flame/cmd/flamectl/resources/schema"
)

const (
	createDesignSchemaCmdArgNum = 2
)

var createDesignSchemaCmd = &cobra.Command{
	Use:   "schema <designId> <schema json file>",
	Short: "Create a new design schema",
	Long:  "This command creates a new design schema",
	Args:  cobra.RangeArgs(createDesignSchemaCmdArgNum, createDesignSchemaCmdArgNum),
	RunE: func(cmd *cobra.Command, args []string) error {
		checkInsecure(cmd)

		designId := args[0]
		schemaPath := args[1]

		params := schema.Params{}
		params.Endpoint = config.ApiServer.Endpoint
		params.User = config.User
		params.DesignId = designId
		params.SchemaPath = schemaPath

		return schema.Create(params)
	},
}

func init() {
	createCmd.AddCommand(createDesignSchemaCmd)
}
