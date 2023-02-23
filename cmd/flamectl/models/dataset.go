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

package models

// DatasetParams provides parameters to generate a new dataset
type DatasetParams struct {
	// CommonParams holds common field params
	CommonParams

	// File is the path of the data file
	File string

	// ID is an optional identification for the dataset
	ID string

	// Limit indicates how many records should be retrieved
	Limit string
}
