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
	"fmt"
	"os"
	"reflect"
	"sort"

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
)

////////////////////////////////////////////////////////////////////////////////
// Job Builder related code
////////////////////////////////////////////////////////////////////////////////

// JobBuilder is a struct used to build a job.
type JobBuilder struct {
	// dbService is the database service used to interact with the database.
	dbService database.DBService
	// jobSpec contains the details of the job to be built.
	jobSpec *openapi.JobSpec
	// jobParams contains the parameters needed to construct the jobSpec.
	jobParams config.JobParams

	// schema contains information for OpenAPI design schema.
	schema openapi.DesignSchema
	// roleCode maps each role to their respective code.
	roleCode map[string][]byte
	// datasets contains a list of available datasets.
	// structure is: trainer role => group name => list of datasets
	datasets map[string]map[string][]openapi.DatasetInfo

	// groupAssociations stores which groups have access to which resources.
	groupAssociations map[string][]map[string]string

	channels map[string]openapi.Channel
}

func NewJobBuilder(dbService database.DBService, jobParams config.JobParams) *JobBuilder {
	jobBuilder := &JobBuilder{
		dbService: dbService,
		jobParams: jobParams,
	}

	jobBuilder.reset()

	return jobBuilder
}

func (b *JobBuilder) GetTasks(jobSpec *openapi.JobSpec) (
	tasks []objects.Task, roles []string, err error,
) {
	b.jobSpec = jobSpec
	if b.jobSpec == nil {
		return nil, nil, fmt.Errorf("job spec is nil")
	}

	b.reset()

	err = b.setup()
	if err != nil {
		return nil, nil, err
	}

	tasks, roles, err = b.build()
	if err != nil {
		return nil, nil, err
	}

	return tasks, roles, nil
}

// reset internal variables of JobBuilder
func (b *JobBuilder) reset() {
	b.roleCode = make(map[string][]byte)
	b.groupAssociations = make(map[string][]map[string]string)
	b.datasets = make(map[string]map[string][]openapi.DatasetInfo)
	b.channels = make(map[string]openapi.Channel)
}

// A function named setup which belongs to the struct JobBuilder is defined, which takes no arguments and returns an error.

func (b *JobBuilder) setup() error {
	// Retrieving data from jobSpec field of the JobBuilder structure.
	spec := b.jobSpec
	// Extracting user ID, design ID, schema version and code version from the JobSpec structure.
	userId, designId, schemaVersion, codeVersion := spec.UserId, spec.DesignId, spec.SchemaVersion, spec.CodeVersion

	// Accessing database service to retrieve the design schema using user ID, design ID and schema version.
	schema, err := b.dbService.GetDesignSchema(userId, designId, schemaVersion)
	if err != nil {
		return err
	}
	// Assigning acquired schema into the schema field of JobBuilder.
	b.schema = schema

	// Getting zipped design code using the user ID, design ID and code version with the help of a database service
	zippedCode, err := b.dbService.GetDesignCode(userId, designId, codeVersion)
	if err != nil {
		return err
	}

	// Create a temporary file with the name util.ProjectName using OS package.
	f, err := os.CreateTemp("", util.ProjectName)
	if err != nil {
		return fmt.Errorf("failed to create temp file: %v", err)
	}
	defer f.Close()

	// Writing zip code into previously created temporary file.
	if _, err = f.Write(zippedCode); err != nil {
		return fmt.Errorf("failed to save zipped code: %v", err)
	}

	// Unzipping extracted zip code information using unzipFile function call and storing files into slice with fdList.
	fdList, err := util.UnzipFile(f)
	if err != nil {
		return fmt.Errorf("failed to unzip file: %v", err)
	}

	// Creating zip code by top level directory from files present in fdList.
	zippedRoleCode, err := util.ZipFileByTopLevelDir(fdList)
	if err != nil {
		return fmt.Errorf("failed to do zip file by top level directory: %v", err)
	}
	// Saving generated zip code by top-level directory in roleCode field of JobBuilder.
	b.roleCode = zippedRoleCode

	// Iterating for each dataset id to fetch dataset info and update the datasets array.
	for roleName, groups := range b.jobSpec.DataSpec.FromSystem {
		if len(groups) == 0 {
			return fmt.Errorf("no dataset group specified for trainer role %s", roleName)
		}

		b.datasets[roleName] = make(map[string][]openapi.DatasetInfo)

		for groupName, datasetIds := range groups {
			if len(datasetIds) == 0 {
				return fmt.Errorf("no dataset specified for trainer role %s, group %s", roleName, groupName)
			}

			for _, datasetId := range datasetIds {
				datasetInfo, err := b.dbService.GetDatasetById(datasetId)
				if err != nil {
					return err
				}

				b.datasets[roleName][groupName] = append(b.datasets[roleName][groupName], datasetInfo)
			}
		}
	}

	for _, role := range b.schema.Roles {
		b.groupAssociations[role.Name] = role.GroupAssociation
	}

	for i, channel := range b.schema.Channels {
		b.channels[channel.Name] = b.schema.Channels[i]
	}

	// Return nil if there are no errors encountered during the execution of declared functions.
	return nil
}

func (b *JobBuilder) build() ([]objects.Task, []string, error) {
	dataRoles, templates := b.getTaskTemplates()
	if err := b.preCheck(dataRoles, templates); err != nil {
		return nil, nil, err
	}

	var tasks []objects.Task

	roleKeys := sortedKeys(templates)
	for _, roleName := range roleKeys {
		tmpl := templates[roleName]

		if !tmpl.isDataConsumer {
			var count int
			for i, associations := range b.groupAssociations[roleName] {
				task := tmpl.Task

				task.ComputeId = util.DefaultRealm
				task.Type = openapi.SYSTEM
				task.Key = util.RandString(taskKeyLen)
				task.JobConfig.GroupAssociation = associations

				index := i + count
				count++

				task.GenerateTaskId(index)

				tasks = append(tasks, task)
			}
			continue
		}

		// TODO: this is absolute and should be removed
		for group, count := range b.jobSpec.DataSpec.FromUser {
			for i := 0; i < int(count); i++ {
				task := tmpl.Task

				task.Type = openapi.USER
				task.JobConfig.Realm = group
				task.JobConfig.GroupAssociation = b.getGroupAssociationByGroup(roleName, group)

				task.GenerateTaskId(i)

				tasks = append(tasks, task)
			}
		}

		var count int
		groups := sortedKeys(b.datasets[roleName])

		for _, groupName := range groups {
			datasets := b.datasets[roleName][groupName]

			for i, dataset := range datasets {
				task := tmpl.Task

				task.ComputeId = dataset.ComputeId
				task.Type = openapi.SYSTEM
				task.Key = util.RandString(taskKeyLen)
				task.JobConfig.DatasetUrl = dataset.Url
				task.JobConfig.GroupAssociation = b.getGroupAssociationByGroup(roleName, groupName)

				index := count + i
				count++

				task.GenerateTaskId(index)

				tasks = append(tasks, task)
			}
		}
	}

	if err := b.postCheck(dataRoles, templates); err != nil {
		return nil, nil, err
	}

	var roles []string

	for _, template := range templates {
		roles = append(roles, template.Role)
	}

	return tasks, roles, nil
}

func (b *JobBuilder) getGroupAssociationByGroup(roleName, groupName string) map[string]string {
	for _, associations := range b.groupAssociations[roleName] {
		for _, association := range associations {
			if association == groupName {
				return associations
			}
		}
	}
	return nil
}

func (b *JobBuilder) getTaskTemplates() ([]string, map[string]*taskTemplate) {
	var dataRoles []string
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

	if err := b.isTemplatesConnected(templates); err != nil {
		// rule 2 violated
		return fmt.Errorf("templates not connected: %s", err.Error())
	}

	// TODO: implement invariant 4

	return nil
}

// isTemplatesConnected function takes a JobBuilder receiver, which will contain the channels and tasks information.
// Additionally, this function takes a map of task templates as arguments.
func (b *JobBuilder) isTemplatesConnected(templates map[string]*taskTemplate) error {
	// A map of string keys and integer values is initialized to keep track of whether a role is found.
	roleFound := make(map[string]int)

	// For each channel in the JobBuilder's channels field
	for _, c := range b.channels {
		// For each role in the Pair field of the current channel
		for _, role := range c.Pair {
			// If the role isn't represented in the task templates passed in as an argument
			if _, ok := templates[role]; !ok {
				// Return an error message saying that the template for the given role wasn't found.
				return fmt.Errorf("template for role %s not found", role)
			}

			// If the role was found in templates map earlier, increment its count
			roleFound[role]++

			// Check number of times a particular role has been referenced across all the channels from the JobBuilder channels record:
			// In case any role is connected to more than 2 roles, it'll throw an error.
			if count, ok := roleFound[role]; ok && count > 2 {
				// Returns an error indicating that the role is related to more than 2 roles.
				return fmt.Errorf("role %s is connected to more than 2 roles", role)
			}
		}
	}

	// For each task in the templates list,
	for _, t := range templates {
		// for each channel associated with the current template.
		for channelName, group := range t.JobConfig.GroupAssociation {
			roleName := t.Role

			// Fetching the channel data
			channel, ok := b.channels[channelName]

			if !ok {
				// If the above line errs out, then this error is returned
				// which says that the channel doesn't exist in the JobBuilder channels.
				return fmt.Errorf("channel %s not found", channelName)
			}

			var found bool

			// Checks if the role of this task exists in the current channel's roles pairing.
			for _, pairRole := range channel.Pair {
				if roleName == pairRole {
					found = true
					break
				}
			}

			if !found {
				// If the role-name associated with the current task template is not a part of the current channel,
				// skip it because rules for it won't apply here.
				continue
			}

			for _, groupBy := range channel.GroupBy.Value {
				if group == groupBy {
					found = true
					break
				}
			}

			if !found {
				// If the 'group' (which belongs to the current task template) isn't found in one of the
				// channels' provided grouping sequence, then we throw this error message.
				return fmt.Errorf("group %s not found in channel %s", group, channelName)
			}
		}
	}

	return nil
	// If no error is thrown during the executiom of this function, then it returns a nil value.
}

func (b *JobBuilder) isConverging(dataRoles []string, templates map[string]*taskTemplate) bool {
	return true
}

func (b *JobBuilder) postCheck(dataRoles []string, templates map[string]*taskTemplate) error {
	// This function will evolve as more invariants are defined
	// At the end of processing templates, the following invariants should be met:
	//

	return nil
}

////////////////////////////////////////////////////////////////////////////////
// Task Template related code
////////////////////////////////////////////////////////////////////////////////

type taskTemplate struct {
	objects.Task

	isDataConsumer bool
}

// This function takes an interface object as input and returns a slice of its string keys in sorted order.
func sortedKeys(obj interface{}) []string {
	v := reflect.ValueOf(obj) // Get the value of the object using reflection

	if v.Kind() == reflect.Map { // Check if it's a map
		mapKeys := v.MapKeys()
		keys := make([]string, len(mapKeys))

		for i := 0; i < len(mapKeys); i++ {
			keys[i] = mapKeys[i].String() // Convert every key to a string
		}

		sort.Strings(keys) // Sort the resulting keys slice

		return keys // return the sorted string slice of keys
	}

	return nil // return the sorted string slice of keys
}
