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

package design

import (
	"github.com/spf13/cobra"

	"github.com/cisco-open/flame/cmd/flamectl/models"
)

func (c *container) GetDesigns() *cobra.Command {
	var command = &cobra.Command{
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

			return c.service.GetMany(&models.DesignParams{
				CommonParams: models.CommonParams{
					Endpoint: c.config.ApiServer.Endpoint,
					User:     c.config.User,
				},
				Limit: limit,
			})
		},
	}

	command.Flags().StringP("limit", "l", "100", "List of all the designs by this user")

	return command
}
