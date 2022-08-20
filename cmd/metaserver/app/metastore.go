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

package app

import (
	"time"

	pbMeta "github.com/cisco-open/flame/pkg/proto/meta"
	"go.uber.org/zap"
)

const (
	HEART_BEAT_DURATION    = 30 * time.Second
	TIMEOUT_STALE_ENDPOINT = 2 * HEART_BEAT_DURATION
)

type job struct {
	channels map[string]*channel
}

type channel struct {
	roles map[string]*metaInfo
}

type metaInfo struct {
	endpoints map[string]chan bool
}

func (j *job) register(mi *pbMeta.MetaInfo) error {
	ch, ok := j.channels[mi.ChName]
	if !ok {
		ch = &channel{roles: make(map[string]*metaInfo)}
		j.channels[mi.ChName] = ch
	}

	if err := ch.register(mi.Me, mi.Endpoint); err != nil {
		return err
	}

	return nil
}

func (j *job) search(chName string, role string) map[string]chan bool {
	ch, ok := j.channels[chName]
	if !ok {
		return nil
	}

	mi := ch.search(role)

	if mi == nil {
		return nil
	}

	return mi.endpoints
}

func (ch *channel) register(role string, endpoint string) error {
	mi, ok := ch.roles[role]
	if !ok {
		mi = &metaInfo{endpoints: make(map[string]chan bool)}
		ch.roles[role] = mi
	}

	_, ok = mi.endpoints[endpoint]
	if ok {
		zap.S().Infof("endpoint %s already registered", endpoint)
		return nil
	}

	// registering for the first time, set heart beat channel nil
	mi.endpoints[endpoint] = nil

	zap.S().Infof("done calling ch.register() for endpoint %s", endpoint)

	return nil
}

func (ch *channel) search(role string) *metaInfo {
	mi, ok := ch.roles[role]
	if !ok {
		return nil
	}

	return mi
}
