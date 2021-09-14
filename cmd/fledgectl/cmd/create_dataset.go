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
	"github.com/spf13/cobra"

	"wwwin-github.cisco.com/eti/fledge/cmd/fledgectl/resources/dataset"
)

var createDatasetCmd = &cobra.Command{
	Use:   "dataset <dataset json file>",
	Short: "Create a new dataset",
	Long:  "This command creates a new dataset",
	Args:  cobra.RangeArgs(1, 1),
	RunE: func(cmd *cobra.Command, args []string) error {
		datasetFile := args[0]

		params := dataset.Params{}
		params.Host = config.ApiServer.Host
		params.Port = config.ApiServer.Port
		params.User = config.User
		params.DatasetFile = datasetFile

		return dataset.Create(params)
	},
}

func init() {
	createCmd.AddCommand(createDatasetCmd)
}