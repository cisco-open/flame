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

	"github.com/cisco-open/flame/cmd/flamectl/resources/job"
)

const (
	flagStatus = "status"
)

var getJobCmd = &cobra.Command{
	Use:   "job <jobId>",
	Short: "Get a job specification or status",
	Long:  "This command retrieves a job specification or status",
	Args:  cobra.RangeArgs(1, 1),
	RunE: func(cmd *cobra.Command, args []string) error {
		checkInsecure(cmd)

		jobId := args[0]

		params := job.Params{}
		params.Endpoint = config.ApiServer.Endpoint
		params.User = config.User
		params.JobId = jobId

		flagStatusValue, _ := cmd.Flags().GetBool(flagStatus)
		if flagStatusValue {
			return job.GetStatus(params)
		}
		return job.Get(params)
	},
}

var getJobsCmd = &cobra.Command{
	Use:   "jobs",
	Short: "Get status of all jobs",
	Long:  "This command retrieves the status of all jobs",
	Args:  cobra.RangeArgs(0, 0),
	RunE: func(cmd *cobra.Command, args []string) error {
		checkInsecure(cmd)

		params := job.Params{}
		params.Endpoint = config.ApiServer.Endpoint
		params.User = config.User

		return job.GetMany(params)
	},
}

func init() {
	getCmd.AddCommand(getJobCmd)
	getJobCmd.Flags().BoolP(flagStatus, "s", false, "Retrieve status of a job")
	getCmd.AddCommand(getJobsCmd)
}
