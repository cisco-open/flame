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
	"fmt"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"

	"github.com/cisco-open/flame/cmd/controller/config"
	"github.com/cisco-open/flame/pkg/openapi"
	"github.com/cisco-open/flame/pkg/util"
)

var (
	testCaseDir = "./testcases/"
)

// TODO maybe use JobBuilder directly?
type TestData struct {
	Dataset openapi.DataSpec     `json:"dataset"`
	Schema  openapi.DesignSchema `json:"schema"`
	JobSpec openapi.JobSpec      `json:"jobSpec"`
	Code    map[string][]byte    `json:"code"`
}

type TestCases struct {
	Filename       string
	TemplatesCount int
	IsTestEnable   bool
}

var testCases = [...]TestCases{
	{
		Filename:       "testcase_hfl.json",
		TemplatesCount: 3,
		IsTestEnable:   true,
	},
	{
		Filename:       "testcase_hfl_with_connected_trainers.json",
		TemplatesCount: -1,
		IsTestEnable:   true,
	},
	{
		Filename:       "testcase_hfl_with_connected_trainers_in_grps.json",
		TemplatesCount: -1,
		IsTestEnable:   true,
	},
	{
		Filename:       "testcase_hfl_with_default_selector.json",
		TemplatesCount: -1,
		IsTestEnable:   true,
	},
	{
		Filename:       "testcase_hfl_with_distributed_selector.json",
		TemplatesCount: -1,
		IsTestEnable:   true,
	},
	{
		Filename:       "testcase_hfl_with_monitoring.json",
		TemplatesCount: -1,
		IsTestEnable:   true,
	},
	{
		Filename:       "testcase_hfl_with_selector_coordinator.json",
		TemplatesCount: -1,
		IsTestEnable:   true,
	},
	{
		Filename:       "testcase_hfl_with_selector_and_monitor.json",
		TemplatesCount: -1,
		IsTestEnable:   true,
	},
	{
		Filename:       "testcase_hfl_diff_midagg.json",
		TemplatesCount: -1,
		IsTestEnable:   true,
	},
	{
		Filename:       "testcase_hfl_diff_midagg_and_selector.json",
		TemplatesCount: -1,
		IsTestEnable:   true,
	},
	{
		Filename:       "testcase_hfl_uneven_depth.json",
		TemplatesCount: -1,
		IsTestEnable:   true,
	},
}

// loadData loads the data for the given test case present in the testcases directory.
func loadData(file string) (TestData, error) {
	testcase := TestData{}
	err := util.ReadFileToStruct(filepath.Join(testCaseDir, file), &testcase)
	if err != nil {
		// use https://jsonlint.com/ to validate the JSON if needed
		fmt.Printf("Error reading the file %v\n", err)
		return testcase, err
	}

	testcase.JobSpec.DataSpec = testcase.Dataset
	testcase.Code = make(map[string][]byte)
	for _, role := range testcase.Schema.Roles {
		testcase.Code[role.Name] = []byte("role code")
	}
	return testcase, nil
}

// only for local testing
func updateZapLogger() {
	zapConfig := zap.NewDevelopmentConfig()
	//custom
	// comment during local testing to display the called name and log level
	zapConfig.EncoderConfig.LevelKey = zapcore.OmitKey
	zapConfig.EncoderConfig.CallerKey = zapcore.OmitKey
	zapConfig.EncoderConfig.EncodeTime = func(t time.Time, enc zapcore.PrimitiveArrayEncoder) {
		enc.AppendString(t.Format("15:04"))
	}
	zapConfig.OutputPaths = []string{"stdout"}
	logger, err := zapConfig.Build()
	if err != nil {
		fmt.Printf("Can't build logger: %v", err)
	}
	defer logger.Sync()
	zap.ReplaceGlobals(logger)
}

func TestBuild(t *testing.T) {
	updateZapLogger()
	for _, tt := range testCases {
		testName := fmt.Sprintf("testFile %s", tt.Filename)
		if tt.IsTestEnable {
			t.Run(testName, func(t *testing.T) {
				builder := NewJobBuilder(nil, config.JobParams{})
				assert.NotNil(t, builder)

				tcData, err := loadData(tt.Filename)
				if err != nil {
					assert.Nil(t, err)
				}
				builder.jobSpec = &tcData.JobSpec
				builder.schema = tcData.Schema
				builder.roleCode = tcData.Code

				_, err = builder.build()
				assert.Nil(t, err)
			})
		}
	}
}
