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
	"fmt"
	"path/filepath"

	"github.com/spf13/cobra"
	"go.uber.org/zap"

	"github.com/cisco-open/flame/cmd/controller/app"
	"github.com/cisco-open/flame/cmd/controller/config"
	"github.com/cisco-open/flame/pkg/util"
)

var (
	cfgFile string
	cfg     *config.Config

	rootCmd = &cobra.Command{
		Use:   util.Controller,
		Short: util.ProjectName + util.Controller,
		RunE: func(cmd *cobra.Command, args []string) error {
			instance, err := app.NewController(cfg)
			if err != nil {
				fmt.Printf("Failed to create a new controller instance\n")
				return nil
			}
			instance.Start()

			return nil
		},
	}
)

func init() {
	cobra.OnInitialize(initConfig)

	usage := "config file (default: /etc/flame/controller.yaml)"
	rootCmd.PersistentFlags().StringVar(&cfgFile, "config", "", usage)
	rootCmd.CompletionOptions.DisableDefaultCmd = true
}

func initConfig() {
	if cfgFile == "" {
		cfgFile = filepath.Join("/etc/flame/controller.yaml")
	}

	var err error

	cfg, err = config.LoadConfig(cfgFile)
	if err != nil {
		zap.S().Fatalf("Failed to load config %s: %v", cfgFile, err)
	}
}

func Execute() error {
	return rootCmd.Execute()
}
