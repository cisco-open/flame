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
	roles map[string]*role
}

type role struct {
	groups map[string]*endInfo
}

type endInfo struct {
	endpoints map[string]chan bool
}

func (j *job) register(mi *pbMeta.MetaInfo) error {
	ch, ok := j.channels[mi.ChName]
	if !ok {
		ch = &channel{roles: make(map[string]*role)}
		j.channels[mi.ChName] = ch
	}

	if err := ch.register(mi); err != nil {
		return err
	}

	return nil
}

func (j *job) search(chName string, roleName string, groupName string) map[string]chan bool {
	ch, ok := j.channels[chName]
	if !ok {
		return nil
	}

	return ch.search(roleName, groupName)
}

func (ch *channel) register(mi *pbMeta.MetaInfo) error {
	myRole, ok := ch.roles[mi.Me]
	if !ok {
		myRole = &role{groups: make(map[string]*endInfo)}
		ch.roles[mi.Me] = myRole
	}

	if err := myRole.register(mi); err != nil {
		return err
	}

	return nil
}

func (ch *channel) search(roleName string, groupName string) map[string]chan bool {
	r, ok := ch.roles[roleName]
	if !ok {
		return nil
	}

	return r.search(groupName)
}

func (r *role) register(mi *pbMeta.MetaInfo) error {
	ei, ok := r.groups[mi.Group]
	if !ok {
		ei = &endInfo{endpoints: make(map[string]chan bool)}
		r.groups[mi.Group] = ei
	}

	ei.register(mi)

	return nil
}

func (r *role) search(groupName string) map[string]chan bool {
	ei, ok := r.groups[groupName]
	if !ok {
		return nil
	}

	return ei.search()
}

func (ei *endInfo) register(mi *pbMeta.MetaInfo) {
	_, ok := ei.endpoints[mi.Endpoint]
	if ok {
		zap.S().Infof("endpoint %s already registered", mi.Endpoint)
	}

	// registering for the first time, set heart beat channel nil
	ei.endpoints[mi.Endpoint] = nil

	zap.S().Infof("done calling ch.register() for endpoint %s", mi.Endpoint)
}

func (ei *endInfo) search() map[string]chan bool {
	return ei.endpoints
}
