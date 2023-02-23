// Copyright 2023 Cisco Systems, Inc. and its affiliates
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

package general

import (
	"github.com/cisco-open/flame/cmd/flamectl/cmd/code"
	"github.com/cisco-open/flame/cmd/flamectl/cmd/dataset"
	"github.com/cisco-open/flame/cmd/flamectl/cmd/design"
	"github.com/cisco-open/flame/cmd/flamectl/cmd/job"
	"github.com/cisco-open/flame/cmd/flamectl/cmd/schema"
	"github.com/cisco-open/flame/cmd/flamectl/models"
	"github.com/samber/do"
	"github.com/spf13/cobra"
)

type IGeneral interface {
	CreateCommands() *cobra.Command
	GetCommands() *cobra.Command
	RemoveComamnds() *cobra.Command
	UpdateCommands() *cobra.Command
	StartCommand() *cobra.Command
	StopCommand() *cobra.Command
}

// container is a struct that stores all the needed dependencies for the commands.
type container struct {
	i      *do.Injector          // i stores the injector that allow to DI the dependencies
	config *models.Configuration // config stores the Configuration struct

	codeCommands    code.ICode
	datasetCommands dataset.IDataset
	designCommands  design.IDesign
	jobCommands     job.IJob
	schemaCommands  schema.ISchema
}

// New creates and returns an instance of the container
func New(i *do.Injector) IGeneral {
	return &container{
		i:               i,
		config:          do.MustInvoke[*models.Configuration](i), // must invoke injects the configuration into the container
		codeCommands:    code.New(i),
		datasetCommands: dataset.New(i),
		designCommands:  design.New(i),
		jobCommands:     job.New(i),
		schemaCommands:  schema.New(i),
	}
}
