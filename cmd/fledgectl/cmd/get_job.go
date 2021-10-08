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
	"fmt"

	"github.com/spf13/cobra"

	"github.com/cisco/fledge/cmd/fledgectl/resources/job"
)

const (
	strJob  = "job"
	strJobs = "jobs"

	maxArgNum4JobStatusCmd = 2
)

var getJobCmd = &cobra.Command{
	Use:   "job <jobId>",
	Short: "Get a job specification",
	Long:  "This command retrieves a job specification",
	Args:  cobra.RangeArgs(1, 1),
	RunE: func(cmd *cobra.Command, args []string) error {
		jobId := args[0]

		params := job.Params{}
		params.Endpoint = config.ApiServer.Endpoint
		params.User = config.User
		params.JobId = jobId

		return job.Get(params)
	},
}

var getJobStatusCmd = &cobra.Command{
	// TODO: maybe need to split status command into sub-commands later
	Use:   "status [job <jobId> | jobs]",
	Short: "Get the status of job(s)",
	Long:  "This command retrieves the status of job(s)",
	Args:  cobra.RangeArgs(1, maxArgNum4JobStatusCmd),
	RunE: func(cmd *cobra.Command, args []string) error {
		resource := args[0]
		if (len(args) == 1 && resource != strJobs) || (len(args) == 2 && resource != strJob) {
			return fmt.Errorf("wrong resource type")
		}

		params := job.Params{}
		params.Endpoint = config.ApiServer.Endpoint
		params.User = config.User

		if resource == strJob {
			jobId := args[1]
			params.JobId = jobId

			return job.GetStatus(params)
		}

		return job.GetStatusMany(params)
	},
}

func init() {
	getCmd.AddCommand(getJobCmd)
	getCmd.AddCommand(getJobStatusCmd)
}
