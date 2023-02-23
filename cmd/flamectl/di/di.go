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

package di

import (
	"github.com/cisco-open/flame/cmd/flamectl/cmd"
	"github.com/cisco-open/flame/cmd/flamectl/models"
	"github.com/cisco-open/flame/pkg/logger"
	"github.com/samber/do"
)

// Container creates a new *do.Injector that can be used to instantiate objects from it
func Container() *do.Injector {
	container := do.New() // Create new injector

	// Providing the functions for generating logger instantiation in the container
	do.Provide(container, logger.New)
	// Providing the functions for generating cmd instantiation in the container
	do.Provide(container, cmd.New)

	// Providing the configuration instance
	do.ProvideValue(container, models.NewConfiguration())

	return container //  Return injector
}
