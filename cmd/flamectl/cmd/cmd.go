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
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"

	"github.com/samber/do"
	"github.com/spf13/cobra"

	"github.com/cisco-open/flame/cmd/flamectl/cmd/general"
	"github.com/cisco-open/flame/cmd/flamectl/models"
	"github.com/cisco-open/flame/pkg/config"
)

const optionInsecure = "insecure"

type ICmd interface {
	Execute() error
	GetCommand() *cobra.Command
}

type cmd struct {
	configFile string
	config     *models.Configuration
	command    *cobra.Command
	injector   *do.Injector
}

/*
 * APPNAME COMMAND ARG --FLAG
 * example hugo server --port=1313 -- 'server' is a command, and 'port' is a flag
 * example codebase https://github.com/schadokar/my-calc
 */
func New(i *do.Injector) (ICmd, error) {
	command := &cobra.Command{
		Use:   "flamectl",
		Short: "flamectl CLI Tool",
		Run: func(cmd *cobra.Command, args []string) {
			cmd.Help()
		},
		RunE: func(cmd *cobra.Command, args []string) error {
			return errors.New("ffffff")
		},
	}

	return &cmd{
		injector: i,
		config:   do.MustInvoke[*models.Configuration](i),
		command:  command,
	}, nil
}

func (c *cmd) Execute() error {
	c.init()

	if err := c.command.Execute(); err != nil {
		return err
	}

	return nil
}

func (c *cmd) init() {
	cobra.OnInitialize(c.initConfig)

	c.initCommands()

	usage := "config file (default: $HOME/.flame/config.yaml)"
	c.command.PersistentFlags().StringVar(&c.configFile, "config", "", usage)
	c.command.PersistentFlags().Bool(optionInsecure, false, "Allow insecure connection")
	c.command.CompletionOptions.DisableDefaultCmd = true
}

func (c *cmd) initCommands() {
	g := general.New(c.injector)

	c.command.AddCommand(
		g.CreateCommands(),
		g.GetCommands(),
		g.UpdateCommands(),
		g.RemoveComamnds(),
		g.StartCommand(),
		g.StopCommand(),
	)
}

func (c *cmd) initConfig() {
	configFile := c.configFile

	if configFile == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			log.Fatalf("Failed to obtain home directory: %v", err)
		}
		configFile = filepath.Join(home, ".flame", "config.yaml")
	}

	// Load the configuration values from external sources
	if err := config.LoadConfigurations(c.config, configFile); err != nil {
		log.Fatalf("Failed to load config %s: %v", configFile, err)
	}

	fmt.Printf("Address of p=&i=%+v:\t%p\n", *c.config, c.config)

	insecureFlag, err := c.command.Flags().GetBool(optionInsecure)
	if err != nil {
		log.Fatalf("Failed to get insecure flag: %s", err.Error())
	} else if insecureFlag {
		fmt.Printf("Warning: --%s flag is set; allow insecure connection.\n", optionInsecure)
		http.DefaultTransport.(*http.Transport).TLSClientConfig = &tls.Config{InsecureSkipVerify: true}
	}
}

func (c *cmd) GetCommand() *cobra.Command {
	return c.command
}
