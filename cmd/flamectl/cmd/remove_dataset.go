// Copyright 2024 Cisco Systems, Inc. and its affiliates
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

package cmd

import (
	"github.com/spf13/cobra"

	"github.com/cisco-open/flame/cmd/flamectl/resources/dataset"
)

var removeDatasetCmd = &cobra.Command{
	Use:   "dataset <datasetId>",
	Short: "Remove a dataset",
	Long:  "This command removes a dataset",
	Args:  cobra.RangeArgs(1, 1),
	RunE: func(cmd *cobra.Command, args []string) error {
		checkInsecure(cmd)

		datasetId := args[0]

		params := dataset.Params{}
		params.Endpoint = config.ApiServer.Endpoint
		params.User = config.User
		params.DatasetId = datasetId

		return dataset.Remove(params)
	},
}

func init() {
	removeCmd.AddCommand(removeDatasetCmd)
}
