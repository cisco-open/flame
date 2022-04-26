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

	"github.com/cisco-open/flame/cmd/flamectl/resources/task"
)

const (
	nArgsGetTask = 2
)

var getTaskCmd = &cobra.Command{
	Use:   "task <jobID> <taskID>",
	Short: "Get the info of a task in a job",
	Long:  "This command retrieves the info of a task in a given job",
	Args:  cobra.RangeArgs(nArgsGetTask, nArgsGetTask),
	RunE: func(cmd *cobra.Command, args []string) error {
		jobId := args[0]
		taskId := args[1]
		params := task.Params{}
		params.Endpoint = config.ApiServer.Endpoint
		params.User = config.User
		params.JobId = jobId
		params.TaskId = taskId

		return task.Get(params)
	},
}

var getTasksCmd = &cobra.Command{
	Use:   "tasks <jobID>",
	Short: "Get info of all tasks in a job",
	Long:  "This command retrieves the info of all tasks in a given job",
	Args:  cobra.RangeArgs(1, 1),
	RunE: func(cmd *cobra.Command, args []string) error {
		jobId := args[0]
		params := task.Params{}
		params.Endpoint = config.ApiServer.Endpoint
		params.User = config.User
		params.JobId = jobId

		return task.GetMany(params)
	},
}

func init() {
	getCmd.AddCommand(getTaskCmd)
	getCmd.AddCommand(getTasksCmd)
}
