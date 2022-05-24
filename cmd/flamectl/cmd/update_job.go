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
	argNum4UpdateJobCmd = 2
)

var updateJobCmd = &cobra.Command{
	Use:   "job <jobId> <job json file>",
	Short: "Update a job specification",
	Long:  "This command updates the specification of a job",
	Args:  cobra.RangeArgs(argNum4UpdateJobCmd, argNum4UpdateJobCmd),
	RunE: func(cmd *cobra.Command, args []string) error {
		checkInsecure(cmd)

		jobId := args[0]
		jobFile := args[1]

		params := job.Params{}
		params.Endpoint = config.ApiServer.Endpoint
		params.User = config.User
		params.JobFile = jobFile
		params.JobId = jobId

		return job.Update(params)
	},
}

func init() {
	updateCmd.AddCommand(updateJobCmd)
}
