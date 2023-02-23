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
// SPDX-License-Identifier: Apache-2.0

package schema

import (
	"github.com/spf13/cobra"

	"github.com/cisco-open/flame/cmd/flamectl/models"
)

func (c *container) RemoveDesignSchema() *cobra.Command {
	var command = &cobra.Command{
		Use:   "schema <version> --design <design id>",
		Short: "Remove a schema with specific version from a design",
		Long:  "This command removes a schema with specific version from a design",
		Args:  cobra.RangeArgs(1, 1),
		RunE: func(cmd *cobra.Command, args []string) error {
			schemaVersion := args[0]

			flags := cmd.Flags()

			designId, err := flags.GetString("design")
			if err != nil {
				return err
			}

			return c.service.Remove(&models.SchemaParams{
				CommonParams: models.CommonParams{
					Endpoint: c.config.ApiServer.Endpoint,
					User:     c.config.User,
				},
				DesignId: designId,
				Version:  schemaVersion,
			})
		},
	}

	command.PersistentFlags().StringP("design", "d", "", "Design ID")
	command.MarkPersistentFlagRequired("design")

	return command
}
