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

package task

import (
	"github.com/spf13/cobra"

	"github.com/cisco-open/flame/cmd/flamectl/models"
)

const nArgsGetTask = 2

func (c *container) GetTask() *cobra.Command {
	var command = &cobra.Command{
		Use:   "task <jobID> <taskID>",
		Short: "Get the info of a task in a job",
		Long:  "This command retrieves the info of a task in a given job",
		Args:  cobra.RangeArgs(nArgsGetTask, nArgsGetTask),
		RunE: func(cmd *cobra.Command, args []string) error {
			jobId := args[0]
			taskId := args[1]

			return c.service.Get(&models.TaskParams{
				CommonParams: models.CommonParams{
					Endpoint: c.config.ApiServer.Endpoint,
					User:     c.config.User,
				},
				Id:    taskId,
				JobId: jobId,
			})
		},
	}

	return command
}
