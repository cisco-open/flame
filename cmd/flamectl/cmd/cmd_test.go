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

package cmd_test

import (
	"testing"

	"github.com/samber/do"
	"github.com/stretchr/testify/assert"

	"github.com/cisco-open/flame/cmd/flamectl/cmd"
	"github.com/cisco-open/flame/cmd/flamectl/models"
	"github.com/cisco-open/flame/pkg/logger"
)

func TestExecuteFailure(t *testing.T) {
	c := setupContainer()

	command, err := cmd.New(c)
	assert.NoError(t, err)

	command.GetCommand().SetArgs([]string{"--unknown_flag"})

	err = command.Execute()
	assert.NotNil(t, err)
	assert.Equal(t, "unknown flag: --unknown_flag", err.Error())
}

func setupContainer() *do.Injector {
	container := do.New()

	do.Provide(container, logger.New)
	do.ProvideValue(container, models.NewConfiguration())

	return container
}
