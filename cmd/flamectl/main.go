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

package main

import (
	"os"

	"github.com/samber/do"
	"go.uber.org/zap"

	"github.com/cisco-open/flame/cmd/flamectl/cmd"
	"github.com/cisco-open/flame/cmd/flamectl/di"
)

func main() {
	// Initialize the Dependency Injection container
	container := di.Container()

	// Get the logger instance
	logger := do.MustInvoke[*zap.Logger](container)

	// Get the list of commands to execute and execute them
	commands := do.MustInvoke[cmd.ICmd](container)

	if err := commands.Execute(); err != nil {
		// If an error occurs, log it and exit with a status code of 1
		logger.Error("error executing command", zap.Error(err))
		os.Exit(1)
	}
}
