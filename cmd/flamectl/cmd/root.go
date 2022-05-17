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
	"crypto/tls"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"

	"github.com/cisco-open/flame/pkg/util"
)

const (
	optionInsecure = "insecure"
)

/*
 * APPNAME COMMAND ARG --FLAG
 * example hugo server --port=1313 -- 'server' is a command, and 'port' is a flag
 * example codebase https://github.com/schadokar/my-calc
 */
var (
	cfgFile string
	config  *Config

	rootCmd = &cobra.Command{
		Use:   util.CliTool,
		Short: util.ProjectName + " CLI Tool",
		Run: func(cmd *cobra.Command, args []string) {
			cmd.Help()
		},
	}
)

/**
 * This is the first function which gets called whenever a package initialize in the golang.
 * https://towardsdatascience.com/how-to-create-a-cli-in-golang-with-cobra-d729641c7177
 * The order of function calls is as follow
 * 	init --> main --> cobra.OnInitialize --> command
 */
func init() {
	cobra.OnInitialize(initConfig)

	usage := "config file (default: $HOME/.flame/config.yaml)"
	rootCmd.PersistentFlags().StringVar(&cfgFile, "config", "", usage)
	rootCmd.PersistentFlags().Bool(optionInsecure, false, "Allow insecure connection")
	rootCmd.CompletionOptions.DisableDefaultCmd = true
}

func initConfig() {
	if cfgFile == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			log.Fatalf("Failed to obtain home directory: %v", err)
		}
		cfgFile = filepath.Join(home, ".flame", "config.yaml")
	}

	var err error

	config, err = loadConfig(cfgFile)
	if err != nil {
		log.Fatalf("Failed to load config %s: %v", cfgFile, err)
	}
}

func Execute() error {
	return rootCmd.Execute()
}

func checkInsecure(cmd *cobra.Command) {
	insecureFlag, _ := cmd.Flags().GetBool(optionInsecure)
	if insecureFlag {
		fmt.Printf("Warning: --%s flag is set; allow insecure connection.\n", optionInsecure)
		http.DefaultTransport.(*http.Transport).TLSClientConfig = &tls.Config{InsecureSkipVerify: true}
	}
}
