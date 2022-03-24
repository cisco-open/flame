// Copyright (c) 2022 Cisco Systems, Inc. and its affiliates
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

	"github.com/cisco-open/flame/cmd/flamectl/resources/task"
)

var getTasksCmd = &cobra.Command{
	Use:   "tasks",
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
	getCmd.AddCommand(getTasksCmd)
}
