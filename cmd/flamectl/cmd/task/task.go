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

package task

import (
	"github.com/samber/do"
	"go.uber.org/zap"

	"github.com/cisco-open/flame/cmd/flamectl/models"
	"github.com/cisco-open/flame/cmd/flamectl/resources/task"
)

// container is a struct used to store configuration pointer
type container struct {
	// config holds an instance of Configuration model
	config  *models.Configuration
	service task.ITask
	logger  *zap.Logger
}

// New initializes a new instance of the container struct with the specified config
func New(i *do.Injector) *container {
	return &container{
		config:  do.MustInvoke[*models.Configuration](i),
		logger:  do.MustInvoke[*zap.Logger](i),
		service: task.New(i),
	}
}