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

package controller

import (
	"wwwin-github.cisco.com/eti/fledge/pkg/openapi"
)

type CacheCollection struct {
	jobAgents map[string][]openapi.ServerInfo
	jobSchema map[string]openapi.DesignSchema
}

var Cache CacheCollection

// TODO: implement functions to get values from the cache.
// This way it would be easy to switch current in-memory implementation with another cache library
func CacheInit() {
	Cache.jobAgents = make(map[string][]openapi.ServerInfo)
	Cache.jobSchema = make(map[string]openapi.DesignSchema)
}
