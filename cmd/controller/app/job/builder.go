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

package job

import (
	"container/list"
	"fmt"
	"io/ioutil"
	"math"
	"strings"

	"wwwin-github.cisco.com/eti/fledge/cmd/controller/app/database"
	"wwwin-github.cisco.com/eti/fledge/cmd/controller/app/objects"
	"wwwin-github.cisco.com/eti/fledge/pkg/openapi"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

const (
	// TODO: for now, hard code the broker url; make this configurable
	broker   = "mqtt.eclipseprojects.io"
	realmSep = "|"
)

////////////////////////////////////////////////////////////////////////////////
// Job Builder related code
////////////////////////////////////////////////////////////////////////////////

type jobBuilder struct {
	jobSpec openapi.JobSpec

	schema   openapi.DesignSchema
	datasets []openapi.DatasetInfo
	roleCode map[string][]byte
}

func newJobBuilder(jobSpec openapi.JobSpec) *jobBuilder {
	return &jobBuilder{jobSpec: jobSpec, datasets: make([]openapi.DatasetInfo, 0)}
}

func (b *jobBuilder) getPayloads() ([]objects.Payload, error) {
	err := b.setup()
	if err != nil {
		return nil, err
	}

	payloads, err := b.build()
	if err != nil {
		return nil, err
	}

	return payloads, nil
}

func (b *jobBuilder) setup() error {
	spec := &b.jobSpec
	userId, designId, codeVersion := spec.UserId, spec.DesignId, spec.CodeVersion

	schema, err := database.GetDesignSchema(userId, designId, codeVersion)
	if err != nil {
		return err
	}
	b.schema = schema

	zippedCode, err := database.GetDesignCode(userId, designId, codeVersion)
	if err != nil {
		return err
	}

	f, err := ioutil.TempFile("", util.ProjectName)
	if err != nil {
		return fmt.Errorf("failed to create temp file: %v", err)
	}
	defer f.Close()

	if _, err = f.Write(zippedCode); err != nil {
		return fmt.Errorf("failed to save zipped code: %v", err)
	}

	fdList, err := util.UnzipFile(f)
	if err != nil {
		return fmt.Errorf("failed to unzip file: %v", err)
	}

	zippedRoleCode, err := util.ZipFileByTopLevelDir(fdList)
	if err != nil {
		return fmt.Errorf("failed to do zip file by top level directory: %v", err)
	}
	b.roleCode = zippedRoleCode

	// update datasets
	for _, datasetId := range b.jobSpec.DatasetIds {
		datasetInfo, err := database.GetDatasetById(datasetId)
		if err != nil {
			return err
		}

		b.datasets = append(b.datasets, datasetInfo)
	}

	return nil
}

func (b *jobBuilder) build() ([]objects.Payload, error) {
	payloads := make([]objects.Payload, 0)

	dataRoles, templates := b.getPayloadTemplates()
	if err := b.preCheck(dataRoles, templates); err != nil {
		return nil, err
	}

	for _, roleName := range dataRoles {
		tmpl, ok := templates[roleName]
		if !ok {
			return nil, fmt.Errorf("failed to locate template for role %s", roleName)
		}

		populated, err := tmpl.walk("", templates, b.datasets)
		if err != nil {
			return nil, err
		}

		payloads = append(payloads, populated...)
	}

	if err := b.postCheck(dataRoles, templates); err != nil {
		return nil, err
	}

	return payloads, nil
}

func (b *jobBuilder) getPayloadTemplates() ([]string, map[string]*payloadTemplate) {
	dataRoles := make([]string, 0)
	templates := make(map[string]*payloadTemplate)

	for _, role := range b.schema.Roles {
		template := &payloadTemplate{}
		JobConfig := &template.JobConfig

		JobConfig.JobId = b.jobSpec.Id
		JobConfig.MaxRunTime = b.jobSpec.MaxRunTime
		JobConfig.BaseModelId = b.jobSpec.BaseModelId
		JobConfig.Hyperparameters = b.jobSpec.Hyperparameters
		JobConfig.Dependencies = b.jobSpec.Dependencies
		JobConfig.BackEnd = string(b.jobSpec.Backend)
		JobConfig.Broker = broker
		// Dataset url will be populated when datasets are handled
		JobConfig.DatasetUrl = ""

		JobConfig.Role = role.Name
		// Realm will be updated when datasets are handled
		JobConfig.Realm = ""
		JobConfig.Channels = b.extractChannels(role.Name, b.schema.Channels)

		template.isDataConsumer = role.IsDataConsumer
		if role.IsDataConsumer {
			dataRoles = append(dataRoles, role.Name)
		}
		template.ZippedCode = b.roleCode[role.Name]

		templates[role.Name] = template
	}

	return dataRoles, templates
}

func (b *jobBuilder) extractChannels(role string, channels []openapi.Channel) []openapi.Channel {
	exChannels := make([]openapi.Channel, 0)

	for _, channel := range channels {
		if contains(channel.Pair, role) {
			exChannels = append(exChannels, channel)
		}
	}

	return exChannels
}

// preCheck checks sanity of templates
func (b *jobBuilder) preCheck(dataRoles []string, templates map[string]*payloadTemplate) error {
	// This function will evolve as more invariants are defined
	// Before processing templates, the following invariants should be met:
	// 1. a role shouled be associated with a code.
	// 2. template should be connected.
	// 3. when graph traversal starts at a data role template, the depth of groupby tag
	//    should strictly decrease from one channel to another.
	// 4. two different data roles cannot be connected directly.

	for _, role := range b.schema.Roles {
		if _, ok := b.roleCode[role.Name]; !ok {
			// rule 1 violated
			return fmt.Errorf("no code found for role %s", role.Name)
		}
	}

	if !b.isTemplatesConnected(templates) {
		// rule 2 violated
		return fmt.Errorf("templates not connected")
	}

	if !b.isConverging(dataRoles, templates) {
		// rule 3 violated
		return fmt.Errorf("groupBy length violated")
	}

	// TODO: implement invariant 4

	return nil
}

func (b *jobBuilder) isTemplatesConnected(templates map[string]*payloadTemplate) bool {
	var start *payloadTemplate
	for _, tmpl := range templates {
		start = tmpl
		break
	}

	if start == nil {
		return true
	}

	start.done = true

	queue := list.New()
	queue.PushBack(start)

	for queue.Len() > 0 {
		elmt := queue.Front()
		tmpl := elmt.Value.(*payloadTemplate)
		// dequeue
		queue.Remove(elmt)

		for _, channel := range tmpl.JobConfig.Channels {
			peerTmpl := tmpl.getPeerTemplate(channel, templates)
			// peer is already visited
			if peerTmpl == nil || peerTmpl.done {
				continue
			}
			peerTmpl.done = true
			queue.PushBack(peerTmpl)
		}
	}

	isConnected := true
	for _, tmpl := range templates {
		if !tmpl.done {
			isConnected = false
		}

		// reset done flag
		tmpl.done = false
	}

	return isConnected
}

func (b *jobBuilder) isConverging(dataRoles []string, templates map[string]*payloadTemplate) bool {
	var start *payloadTemplate
	for _, tmpl := range templates {
		start = tmpl
		break
	}

	if start == nil {
		return true
	}

	ruleSatisfied := true
	for _, dataRole := range dataRoles {
		tmpl := templates[dataRole]
		tmpl.done = true
		for _, channel := range tmpl.JobConfig.Channels {
			peerTmpl := tmpl.getPeerTemplate(channel, templates)
			if peerTmpl == nil || peerTmpl.done || peerTmpl.isDataConsumer {
				continue
			}

			tmpl.minGroupByTokenLen = math.MaxInt32
			if !_walkForGroupByCheck(templates, tmpl, peerTmpl) {
				ruleSatisfied = false
				break
			}
		}
	}

	for _, tmpl := range templates {
		// reset done flag
		tmpl.done = false
	}

	return ruleSatisfied
}

func (b *jobBuilder) postCheck(dataRoles []string, templates map[string]*payloadTemplate) error {
	// This function will evolve as more invariants are defined
	// At the end of processing templates, the following invariants should be met:
	//

	return nil
}

// _walkForGroupByCheck determines if the convergence rule is violated or not; this method shouldn't be called directly
func _walkForGroupByCheck(templates map[string]*payloadTemplate, prevTmpl *payloadTemplate, tmpl *payloadTemplate) bool {
	funcMin := func(a int, b int) int {
		if a < b {
			return a
		}
		return b
	}

	tmpl.done = true

	// compute minLen of groupBy field for already visited roles
	for _, channel := range tmpl.JobConfig.Channels {
		peerTmpl := tmpl.getPeerTemplate(channel, templates)
		if prevTmpl != peerTmpl {
			continue
		}

		minLen := 0
		tmpLen := math.MaxInt32
		for _, val := range channel.GroupBy.Value {
			length := len(strings.Split(val, realmSep))
			tmpLen = funcMin(tmpLen, length)
		}

		if tmpLen < math.MaxInt32 {
			minLen = tmpLen
		}

		if prevTmpl.minGroupByTokenLen > 0 && minLen >= prevTmpl.minGroupByTokenLen {
			// rule violation detected
			return false
		}

		tmpl.minGroupByTokenLen = minLen
		break
	}

	for _, channel := range tmpl.JobConfig.Channels {
		peerTmpl := tmpl.getPeerTemplate(channel, templates)
		if peerTmpl == nil || peerTmpl.done || peerTmpl.isDataConsumer || prevTmpl == peerTmpl {
			continue
		}

		if !_walkForGroupByCheck(templates, tmpl, peerTmpl) {
			return false
		}
	}

	return true
}

////////////////////////////////////////////////////////////////////////////////
// Payload Template related code
////////////////////////////////////////////////////////////////////////////////

type payloadTemplate struct {
	objects.Payload

	isDataConsumer bool

	done bool

	minGroupByTokenLen int
}

func (tmpl *payloadTemplate) getPeerTemplate(channel openapi.Channel, templates map[string]*payloadTemplate) *payloadTemplate {
	if !contains(channel.Pair, tmpl.JobConfig.Role) {
		return nil
	}

	peer := channel.Pair[0]
	if tmpl.JobConfig.Role == peer {
		peer = channel.Pair[1]
	}

	peerTmpl, ok := templates[peer]
	if !ok {
		return nil
	}

	return peerTmpl
}

func (tmpl *payloadTemplate) walk(prevPeer string, templates map[string]*payloadTemplate,
	datasets []openapi.DatasetInfo) ([]objects.Payload, error) {
	payloads := make([]objects.Payload, 0)

	populated := tmpl.buildPayloads(prevPeer, templates, datasets)
	if len(populated) > 0 {
		payloads = append(payloads, populated...)
	}

	// if template is not for data consumer role
	for _, channel := range tmpl.JobConfig.Channels {
		peerTmpl := tmpl.getPeerTemplate(channel, templates)
		// peer is already handled
		if peerTmpl == nil || peerTmpl.done {
			continue
		}

		populated, err := peerTmpl.walk(tmpl.JobConfig.Role, templates, datasets)
		if err != nil {
			return nil, fmt.Errorf("failed to populdate template")
		}

		payloads = append(payloads, populated...)
	}

	return payloads, nil
}

// buildPayloads returns an array of Payload generated from template; this function should be called via walk()
func (tmpl *payloadTemplate) buildPayloads(prevPeer string, templates map[string]*payloadTemplate,
	datasets []openapi.DatasetInfo) []objects.Payload {
	payloads := make([]objects.Payload, 0)

	defer func() {
		// handling this template is done
		tmpl.done = true
	}()

	// in case of data consumer template
	if tmpl.isDataConsumer {
		// TODO: currently, one data role is assumed; therefore, datasets are used for one data role.
		//       to support more than one data role, datasets should be associated with each role.
		//       this needs job spec modification.
		for i, dataset := range datasets {
			payload := tmpl.Payload
			payload.JobConfig.DatasetUrl = dataset.Url
			payload.JobConfig.Realm = dataset.Realm
			// no need to copy byte array; assignment suffices
			payload.ZippedCode = tmpl.Payload.ZippedCode
			payload.JobId = payload.JobConfig.JobId
			payload.GenerateAgentId(i)

			payloads = append(payloads, payload)
		}

		return payloads
	}

	prevTmpl := templates[prevPeer]
	for _, channel := range prevTmpl.JobConfig.Channels {
		if !contains(channel.Pair, tmpl.JobConfig.Role) {
			continue
		}

		i := 0
		for i < len(channel.GroupBy.Value) {
			payload := tmpl.Payload
			payload.JobId = payload.JobConfig.JobId
			payload.JobConfig.Realm = channel.GroupBy.Value[i] + realmSep + util.ProjectName
			payload.GenerateAgentId(i)

			payloads = append(payloads, payload)
			i++
		}

		// no payload is added to payloads (payload array),
		// which means that groupby is not specified; so,
		// we have to create a default payload
		if len(payloads) == 0 {
			payload := tmpl.Payload
			payload.JobId = payload.JobConfig.JobId
			payload.GenerateAgentId(0)

			payloads = append(payloads, payload)
		}
	}

	return payloads
}

////////////////////////////////////////////////////////////////////////////////
// Etc
////////////////////////////////////////////////////////////////////////////////

func contains(haystack []string, needle string) bool {
	for i := range haystack {
		if needle == haystack[i] {
			return true
		}
	}

	return false
}
