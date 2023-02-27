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

package job

import (
	"github.com/cisco-open/flame/cmd/flamectl/models"
	"github.com/samber/do"
	"go.uber.org/zap"
)

type IJob interface {
	Create(params *models.JobParams) error
	Get(params *models.JobParams) error
	GetMany(params *models.JobParams) error
	Update(params *models.JobParams) error
	Remove(params *models.JobParams) error
	Start(params *models.JobParams) error
	Stop(params *models.JobParams) error
	GetStatus(params *models.JobParams) error
}

type container struct {
	logger *zap.Logger
}

// New initializes a new instance of the container struct with the specified config
func New(i *do.Injector) IJob {
	return &container{
		logger: do.MustInvoke[*zap.Logger](i),
	}
}
