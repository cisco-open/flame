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

	"github.com/cisco/fledge/cmd/controller/app/database"
	"github.com/cisco/fledge/cmd/controller/app/objects"
	"github.com/cisco/fledge/cmd/controller/config"
	"github.com/cisco/fledge/pkg/openapi"
	"github.com/cisco/fledge/pkg/util"
)

const (
	realmSep = "|"
)

////////////////////////////////////////////////////////////////////////////////
// Job Builder related code
////////////////////////////////////////////////////////////////////////////////

type jobBuilder struct {
	dbService database.DBService
	jobSpec   openapi.JobSpec
	brokers   []config.Broker
	registry  config.Registry

	schema   openapi.DesignSchema
	datasets []openapi.DatasetInfo
	roleCode map[string][]byte
}

func newJobBuilder(dbService database.DBService, jobSpec openapi.JobSpec, brokers []config.Broker, registry config.Registry) *jobBuilder {
	return &jobBuilder{
		dbService: dbService,
		jobSpec:   jobSpec,
		brokers:   brokers,
		registry:  registry,
		datasets:  make([]openapi.DatasetInfo, 0),
	}
}

func (b *jobBuilder) getTasks() ([]objects.Task, []string, error) {
	err := b.setup()
	if err != nil {
		return nil, nil, err
	}

	tasks, roles, err := b.build()
	if err != nil {
		return nil, nil, err
	}

	return tasks, roles, nil
}

func (b *jobBuilder) setup() error {
	spec := &b.jobSpec
	userId, designId, schemaVersion, codeVersion := spec.UserId, spec.DesignId, spec.SchemaVersion, spec.CodeVersion

	schema, err := b.dbService.GetDesignSchema(userId, designId, schemaVersion)
	if err != nil {
		return err
	}
	b.schema = schema

	zippedCode, err := b.dbService.GetDesignCode(userId, designId, codeVersion)
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
		datasetInfo, err := b.dbService.GetDatasetById(datasetId)
		if err != nil {
			return err
		}

		b.datasets = append(b.datasets, datasetInfo)
	}

	return nil
}

func (b *jobBuilder) build() ([]objects.Task, []string, error) {
	tasks := make([]objects.Task, 0)
	roles := make([]string, 0)

	dataRoles, templates := b.getTaskTemplates()
	if err := b.preCheck(dataRoles, templates); err != nil {
		return nil, nil, err
	}

	for _, roleName := range dataRoles {
		tmpl, ok := templates[roleName]
		if !ok {
			return nil, nil, fmt.Errorf("failed to locate template for role %s", roleName)
		}

		populated, err := tmpl.walk("", templates, b.datasets)
		if err != nil {
			return nil, nil, err
		}

		tasks = append(tasks, populated...)
	}

	if err := b.postCheck(dataRoles, templates); err != nil {
		return nil, nil, err
	}

	for _, template := range templates {
		roles = append(roles, template.Role)
	}

	return tasks, roles, nil
}

func (b *jobBuilder) getTaskTemplates() ([]string, map[string]*taskTemplate) {
	dataRoles := make([]string, 0)
	templates := make(map[string]*taskTemplate)

	for _, role := range b.schema.Roles {
		template := &taskTemplate{}
		JobConfig := &template.JobConfig

		JobConfig.Job.Id = b.jobSpec.Id
		// DesignId is a string suitable as job's name
		JobConfig.Job.Name = b.jobSpec.DesignId
		JobConfig.MaxRunTime = b.jobSpec.MaxRunTime
		JobConfig.BaseModel = b.jobSpec.BaseModel
		JobConfig.Hyperparameters = b.jobSpec.Hyperparameters
		JobConfig.Dependencies = b.jobSpec.Dependencies
		JobConfig.BackEnd = string(b.jobSpec.Backend)
		JobConfig.Brokers = b.brokers
		JobConfig.Registry = b.registry
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
		template.Role = role.Name

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
func (b *jobBuilder) preCheck(dataRoles []string, templates map[string]*taskTemplate) error {
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

func (b *jobBuilder) isTemplatesConnected(templates map[string]*taskTemplate) bool {
	var start *taskTemplate
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
		tmpl := elmt.Value.(*taskTemplate)
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

func (b *jobBuilder) isConverging(dataRoles []string, templates map[string]*taskTemplate) bool {
	var start *taskTemplate
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

func (b *jobBuilder) postCheck(dataRoles []string, templates map[string]*taskTemplate) error {
	// This function will evolve as more invariants are defined
	// At the end of processing templates, the following invariants should be met:
	//

	return nil
}

// _walkForGroupByCheck determines if the convergence rule is violated or not; this method shouldn't be called directly
func _walkForGroupByCheck(templates map[string]*taskTemplate, prevTmpl *taskTemplate, tmpl *taskTemplate) bool {
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
// Task Template related code
////////////////////////////////////////////////////////////////////////////////

type taskTemplate struct {
	objects.Task

	isDataConsumer bool

	done bool

	minGroupByTokenLen int
}

func (tmpl *taskTemplate) getPeerTemplate(channel openapi.Channel, templates map[string]*taskTemplate) *taskTemplate {
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

func (tmpl *taskTemplate) walk(prevPeer string, templates map[string]*taskTemplate,
	datasets []openapi.DatasetInfo) ([]objects.Task, error) {
	tasks := make([]objects.Task, 0)

	populated := tmpl.buildTasks(prevPeer, templates, datasets)
	if len(populated) > 0 {
		tasks = append(tasks, populated...)
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

		tasks = append(tasks, populated...)
	}

	return tasks, nil
}

// buildTasks returns an array of Task generated from template; this function should be called via walk()
func (tmpl *taskTemplate) buildTasks(prevPeer string, templates map[string]*taskTemplate,
	datasets []openapi.DatasetInfo) []objects.Task {
	tasks := make([]objects.Task, 0)

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
			task := tmpl.Task
			task.JobConfig.DatasetUrl = dataset.Url
			task.JobConfig.Realm = dataset.Realm
			// no need to copy byte array; assignment suffices
			task.ZippedCode = tmpl.Task.ZippedCode
			task.JobId = task.JobConfig.Job.Id
			task.GenerateAgentId(i)

			tasks = append(tasks, task)
		}

		return tasks
	}

	prevTmpl := templates[prevPeer]
	for _, channel := range prevTmpl.JobConfig.Channels {
		if !contains(channel.Pair, tmpl.JobConfig.Role) {
			continue
		}

		i := 0
		for i < len(channel.GroupBy.Value) {
			task := tmpl.Task
			task.JobId = task.JobConfig.Job.Id
			task.JobConfig.Realm = channel.GroupBy.Value[i] + realmSep + util.ProjectName
			task.GenerateAgentId(i)

			tasks = append(tasks, task)
			i++
		}

		// no task is added to tasks (task array),
		// which means that groupby is not specified; so,
		// we have to create a default task
		if len(tasks) == 0 {
			task := tmpl.Task
			task.JobId = task.JobConfig.Job.Id
			task.GenerateAgentId(0)

			tasks = append(tasks, task)
		}
	}

	return tasks
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
