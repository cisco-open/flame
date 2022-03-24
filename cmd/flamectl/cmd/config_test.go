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
	"testing"

	"github.com/spf13/afero"
	"github.com/stretchr/testify/assert"

	"github.com/cisco-open/flame/pkg/util"
)

var (
	testData = `
apiserver:
  endpoint: localhost:10100
user: john
`
)

func TestLoadConfig(t *testing.T) {
	fs = afero.NewMemMapFs()

	configFilePath := "/testConfig"

	config, err := loadConfig(configFilePath)
	assert.NotNil(t, err)
	assert.Nil(t, config)

	// write test data
	err = afero.WriteFile(fs, configFilePath, []byte(testData), util.FilePerm0644)
	assert.Nil(t, err)

	config, err = loadConfig(configFilePath)
	assert.Nil(t, err)
	assert.Equal(t, "localhost:10100", config.ApiServer.Endpoint)
	assert.Equal(t, "john", config.User)
}
