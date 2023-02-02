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

package job

import (
	"container/list"
	"fmt"
	"math"
	"os"
	"strings"

	"github.com/cisco-open/flame/cmd/controller/app/database"
	"github.com/cisco-open/flame/cmd/controller/app/objects"
	"github.com/cisco-open/flame/cmd/controller/config"
	"github.com/cisco-open/flame/pkg/openapi"
	"github.com/cisco-open/flame/pkg/util"
)

const (
	defaultGroup   = "default"
	groupByTypeTag = "tag"
	taskKeyLen     = 32

	emptyTaskKey    = ""
	emptyDatasetUrl = ""
)

////////////////////////////////////////////////////////////////////////////////
// Job Builder related code
////////////////////////////////////////////////////////////////////////////////

type JobBuilder struct {
	dbService database.DBService
	jobSpec   *openapi.JobSpec
	jobParams config.JobParams

	schema   openapi.DesignSchema
	datasets []openapi.DatasetInfo
	roleCode map[string][]byte
}

func NewJobBuilder(dbService database.DBService, jobParams config.JobParams) *JobBuilder {
	return &JobBuilder{
		dbService: dbService,
		jobParams: jobParams,
		datasets:  make([]openapi.DatasetInfo, 0),
	}
}

func (b *JobBuilder) GetTasks(jobSpec *openapi.JobSpec) ([]objects.Task, []string, error) {
	b.jobSpec = jobSpec
	if b.jobSpec == nil {
		return nil, nil, fmt.Errorf("job spec is nil")
	}

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

func (b *JobBuilder) setup() error {
	spec := b.jobSpec
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

	f, err := os.CreateTemp("", util.ProjectName)
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

	// reset datasets array
	b.datasets = make([]openapi.DatasetInfo, 0)
	// update datasets
	for _, datasetId := range b.jobSpec.DataSpec.FromSystem {
		datasetInfo, err := b.dbService.GetDatasetById(datasetId)
		if err != nil {
			return err
		}

		b.datasets = append(b.datasets, datasetInfo)
	}

	return nil
}

func (b *JobBuilder) build() ([]objects.Task, []string, error) {
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

		populated, err := tmpl.walk("", templates, b.datasets, b.jobSpec.DataSpec.FromUser)
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

func (b *JobBuilder) getTaskTemplates() ([]string, map[string]*taskTemplate) {
	dataRoles := make([]string, 0)
	templates := make(map[string]*taskTemplate)

	for _, role := range b.schema.Roles {
		template := &taskTemplate{}
		jobConfig := &template.JobConfig

		jobConfig.Configure(b.jobSpec, b.jobParams.Brokers, b.jobParams.Registry, role, b.schema.Channels)

		// check channels and set default group if channels don't have groupBy attributes set
		for i := range jobConfig.Channels {
			if len(jobConfig.Channels[i].GroupBy.Value) > 0 {
				continue
			}

			// since there is no groupBy attribute, set default
			jobConfig.Channels[i].GroupBy.Type = groupByTypeTag
			jobConfig.Channels[i].GroupBy.Value = append(jobConfig.Channels[i].GroupBy.Value, defaultGroup)
		}

		template.isDataConsumer = role.IsDataConsumer
		if role.IsDataConsumer {
			dataRoles = append(dataRoles, role.Name)
		}
		template.ZippedCode = b.roleCode[role.Name]
		template.Role = role.Name
		template.JobId = jobConfig.Job.Id

		templates[role.Name] = template
	}

	return dataRoles, templates
}

// preCheck checks sanity of templates
func (b *JobBuilder) preCheck(dataRoles []string, templates map[string]*taskTemplate) error {
	// This function will evolve as more invariants are defined
	// Before processing templates, the following invariants should be met:
	// 1. At least one data consumer role should be defined.
	// 2. a role should be associated with a code.
	// 3. template should be connected.
	// 4. when graph traversal starts at a data role template, the depth of groupBy tag
	//    should strictly decrease from one channel to another.
	// 5. two different data roles cannot be connected directly.

	if len(dataRoles) == 0 {
		return fmt.Errorf("no data consumer role found")
	}

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

func (b *JobBuilder) isTemplatesConnected(templates map[string]*taskTemplate) bool {
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

func (b *JobBuilder) isConverging(dataRoles []string, templates map[string]*taskTemplate) bool {
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

func (b *JobBuilder) postCheck(dataRoles []string, templates map[string]*taskTemplate) error {
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
			length := len(strings.Split(val, util.RealmSep))
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
	if !util.Contains(channel.Pair, tmpl.JobConfig.Role) {
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
	datasets []openapi.DatasetInfo, userDatasetKV map[string]int32) ([]objects.Task, error) {
	tasks := make([]objects.Task, 0)

	populated := tmpl.buildTasks(prevPeer, templates, datasets, userDatasetKV)
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

		populated, err := peerTmpl.walk(tmpl.JobConfig.Role, templates, datasets, userDatasetKV)
		if err != nil {
			return nil, fmt.Errorf("failed to populdate template")
		}

		tasks = append(tasks, populated...)
	}

	return tasks, nil
}

// buildTasks returns an array of Task generated from template; this function should be called via walk()
func (tmpl *taskTemplate) buildTasks(prevPeer string, templates map[string]*taskTemplate,
	datasets []openapi.DatasetInfo, userDatasetKV map[string]int32) []objects.Task {
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
		for group, count := range userDatasetKV {
			for i := 0; i < int(count); i++ {
				task := tmpl.Task
				task.Configure(openapi.USER, emptyTaskKey, group, emptyDatasetUrl, i)
				tasks = append(tasks, task)
			}
		}

		for i, dataset := range datasets {
			task := tmpl.Task
			task.ComputeId = dataset.ComputeId
			task.Configure(openapi.SYSTEM, util.RandString(taskKeyLen), dataset.Realm, dataset.Url, i)
			tasks = append(tasks, task)
		}

		return tasks
	}

	prevTmpl := templates[prevPeer]
	for _, channel := range prevTmpl.JobConfig.Channels {
		if !util.Contains(channel.Pair, tmpl.JobConfig.Role) {
			continue
		}

		for i := 0; i < len(channel.GroupBy.Value); i++ {
			task := tmpl.Task
			realm := channel.GroupBy.Value[i] + util.RealmSep + util.ProjectName
			task.ComputeId = util.DefaultRealm
			task.Configure(openapi.SYSTEM, util.RandString(taskKeyLen), realm, emptyDatasetUrl, i)

			tasks = append(tasks, task)
		}
	}

	return tasks
}
