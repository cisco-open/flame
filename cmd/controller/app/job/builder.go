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

	"go.uber.org/zap"

	"github.com/cisco-open/flame/cmd/controller/app/database"
	"github.com/cisco-open/flame/cmd/controller/app/objects"
	"github.com/cisco-open/flame/cmd/controller/config"
	"github.com/cisco-open/flame/pkg/openapi"
	"github.com/cisco-open/flame/pkg/util"
)

const (
	defaultGroup    = "default"
	groupByTypeTag  = "tag"
	taskKeyLen      = 32
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
	roleCode map[string][]byte
}

// roleInfo extracts the basic information associated with the role that is required during TAG expansion
type roleInfo struct {
	Role     openapi.Role
	Channels []openapi.Channel
}

// nodeCollection is collection of tasks associated with a role as TAG is expanded
type nodeCollection struct {
	Tasks      []*objects.Task
	IsExpanded bool
	IsVisited  bool
	Role       string
}

// tagExpander holds information required to expand a given TAG
type tagExpander struct {
	JobBuilder        JobBuilder
	RoleInfo          map[string]roleInfo //collection to store role information associated with all the roles in the given TAG
	DataConsumerRoles []string
	TopologyMap       map[string]*nodeCollection // collection of list of tasks associated with a role. Populated during TAG expansion,
	VisitedChannels   map[string]bool
	ConnectedRoleQ    []string //Q of roles to be explored during TAG expansion
}

func NewJobBuilder(dbService database.DBService, jobParams config.JobParams) *JobBuilder {
	return &JobBuilder{
		dbService: dbService,
		jobParams: jobParams,
	}
}

func (b *JobBuilder) GetTasks(jobSpec *openapi.JobSpec) (map[string][]*objects.Task, []string, error) {
	b.jobSpec = jobSpec
	if b.jobSpec == nil {
		return nil, nil, fmt.Errorf("job spec is nil")
	}

	err := b.setup()
	if err != nil {
		return nil, nil, err
	}

	tagExpObj, err := b.build()
	if err != nil {
		return nil, nil, err
	}

	roles := make([]string, 0)
	topologyTasks := map[string][]*objects.Task{}

	for roleName, value := range tagExpObj.TopologyMap {
		roles = append(roles, roleName)
		topologyTasks[roleName] = value.Tasks
	}

	return topologyTasks, roles, nil
}

func (b *JobBuilder) setup() error {
	spec := b.jobSpec
	userId, designId, schemaVersion, codeVersion := spec.UserId, spec.DesignId, spec.SchemaVersion, spec.CodeVersion

	// TODO should not manipulate global jobBuilder object for every request.
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

	return nil
}

// //////////////////////////////////////////////////////////////////////////////
// TAG Expansion Logic
// //////////////////////////////////////////////////////////////////////////////

// build is responsible to pre-check and expand the TAG associated with a job. Thus, creating all the associated tasks for each role.
func (b *JobBuilder) build() (*tagExpander, error) {
	tagExpanderObj := tagExpander{}
	tagExpanderObj.JobBuilder = *b
	tagExpanderObj.getRoleInfo()

	tagExpanderObj.TopologyMap = map[string]*nodeCollection{}
	tagExpanderObj.VisitedChannels = map[string]bool{}

	for role, _ := range tagExpanderObj.RoleInfo {
		tagExpanderObj.TopologyMap[role] = &nodeCollection{
			Tasks:      make([]*objects.Task, 0),
			Role:       role,
			IsExpanded: false,
			IsVisited:  false,
		}
	}

	if err := tagExpanderObj.preCheck(); err != nil {
		return nil, err
	}

	if err := tagExpanderObj.expandTAG(); err != nil {
		zap.S().Errorf("TAG expansion for job %s failed. %v", b.jobSpec.Id, err)
		return nil, err
	}

	if err := tagExpanderObj.postCheck(); err != nil {
		return nil, err
	}

	return &tagExpanderObj, nil
}

// preCheck checks sanity of templates
func (t *tagExpander) preCheck() error {
	// This function will evolve as more invariants are defined
	// Before processing templates, the following invariants should be met:
	// 1. At least one data consumer role should be defined.
	// 2. a role should be associated with a code.

	if len(t.DataConsumerRoles) == 0 {
		return fmt.Errorf("no data consumer role found")
	}

	for _, role := range t.JobBuilder.schema.Roles {
		if _, ok := t.JobBuilder.roleCode[role.Name]; !ok {
			// rule 1 violated
			return fmt.Errorf("no code found for role %s", role.Name)
		}
	}

	// TODO: implement other invariant
	return nil
}

func (t *tagExpander) postCheck() error {
	// This function will evolve as more invariants are defined
	// At the end of processing templates, the following invariants should be met:
	return nil
}

// expandTAG expands the TAG representation to physical deployment graph using BFS technique.
// It iterates dataConsumer role and visits exploreRole to explore all the channels associated with the given role.
// As a role is explored ConnectedRoleQ maintains the list of peer roles connected with the current role provided to the exploreRole function.
// After exploring all the currRole edges, each peerRole is visited one by one in similar manner.
func (t *tagExpander) expandTAG() error {
	t.ConnectedRoleQ = make([]string, 0)
	for _, dataRole := range t.DataConsumerRoles {
		zap.S().Debugf("exploring datarole %s\n", dataRole)
		//if either role is a DataConsumer need to ensure trainer nodes are created before proceeding to further exploration
		err := t.buildDataWorkers(dataRole)
		if err != nil {
			zap.S().Errorf("TAG expansion failed %v", err)
			return err
		}

		err = t.exploreRole(dataRole)
		if err != nil {
			zap.S().Errorf("TAG expansion failed, happened during dfsWalk for datarole %s. %v", dataRole, err)
			return err
		}
	}

	// BFS style visiting all the connected components while Q is not empty
	Qsize := len(t.ConnectedRoleQ)
	zap.S().Debugf("ConnectedRoleQ %v", t.ConnectedRoleQ)
	for Qsize > 0 {
		err := t.exploreRole(t.ConnectedRoleQ[0])
		if err != nil {
			return err
		}
		t.ConnectedRoleQ = t.ConnectedRoleQ[1:] //dequeue
		Qsize = len(t.ConnectedRoleQ)
		zap.S().Debugf("ConnectedRoleQ %v", t.ConnectedRoleQ)
	}

	t.printTopology()
	return nil
}

// exploreRole visits all the channels associated with the given role. As channels are explored the connected peerRole tasks/nodes are either created or connected tasks are updated.
// After processing a channel corresponding peer role is added to the Q and is explored in the future.
func (t *tagExpander) exploreRole(currRole string) error {
	zap.S().Debugf("explore currRole %s. Is visited %v and current nodes %d", currRole, t.TopologyMap[currRole].IsVisited, len(t.TopologyMap[currRole].Tasks))

	if t.TopologyMap[currRole].IsExpanded {
		zap.S().Debugf("role %s is already expanded", currRole)
		return nil
	}

	currTmpl := t.RoleInfo[currRole]
	for _, channel := range currTmpl.Channels {
		peerTmpl, err := getPeerInfo(currRole, channel, t.RoleInfo)
		if err != nil {
			zap.S().Errorf("failed to determine peer information")
			return err
		}
		peerRole := peerTmpl.Role.Name
		zap.S().Debugf("explore channel %s between currRole %s and peerRole %s", channel.Name, currRole, peerRole)

		// is the edge is already explored, skip!
		key := fmt.Sprintf("%s:%s", currRole, peerRole)
		revKey := fmt.Sprintf("%s:%s", peerRole, currRole)
		if t.VisitedChannels[key] || t.VisitedChannels[revKey] {
			zap.S().Debugf("channel %s already visited", channel.Name)
			continue
		}

		// maintain the following order
		// 1. check if self connection
		// 2. check isVisited
		switch peerNodeColl := t.TopologyMap[peerRole]; {
		case peerRole == currRole:
			// self channel => every node in the currRole connects to itself
			err = t.exploreSelfConnectingChannel(currRole, channel)
		case peerNodeColl.IsVisited == false:
			// peerRole is visited for the first time => require creation of new nodes/task
			err = t.exploreNewPeerRoleChannel(currRole, peerRole, channel)
		case peerNodeColl.IsVisited:
			// peerRole is already visited through another channel. Would require updating the label and connectedTaskList
			err = t.exploreVisitedPeerRoleChannel(currRole, peerRole, channel)
		}

		if err != nil {
			zap.S().Errorf("TAG expansion failed while exploring channel %s between (currRole) %s -- %s (peerRole). %v", channel.Name, currRole, peerRole, err)
			return err
		}

		t.VisitedChannels[key] = true
		t.ConnectedRoleQ = append(t.ConnectedRoleQ, peerRole)
	}

	//visited connected roles and expanded the edges/channels connected to the currRole
	t.TopologyMap[currRole].IsExpanded = true
	zap.S().Debugf("DONE with exploreRole for currRole %s", currRole)
	return nil
}

// exploreSelfConnectingChannel for a given role it creates tasks connection within its tasks/nodes.
func (t *tagExpander) exploreSelfConnectingChannel(currRole string, channel openapi.Channel) error {
	zap.S().Debugf("self connected channel %s found ", channel.Name)
	interConnect := func(tasksList []*objects.Task) {
		if len(tasksList) < 2 {
			zap.S().Debugf("ignoring self connection of single task")
			return
		}
		for _, rootTsk := range tasksList {
			zap.S().Debugf("interconnected nodes taskId:%s | label:%s)", rootTsk.TaskId, rootTsk.Label)
			//fmt.Printf(" node (id:%s | label:%s)", rootTsk.Label, rootTsk.TaskId)
			for _, childTsk := range tasksList {
				if rootTsk.TaskId != childTsk.TaskId {
					rootTsk.ConnectedTaskIds[currRole] = append(rootTsk.ConnectedTaskIds[currRole], childTsk.TaskId)
				}
			}
		}
	}

	switch grpValues := channel.GroupBy.Value; {
	case containsInNestedArray(grpValues, util.DefaultRealm) == true:
		// if groupByValue is default, means every node is connected to itself
		interConnect(t.TopologyMap[currRole].Tasks)
	default:
		// group them based on the given list.
		for _, valueList := range grpValues {
			// if grpByValue is in task.label select the task
			tmpTasks := make([]*objects.Task, 0)
			for _, l := range valueList {
				taskFound := false
				for _, tmp := range t.TopologyMap[currRole].Tasks {
					for _, tmpLabel := range tmp.Label {
						if l == tmpLabel {
							tmpTasks = append(tmpTasks, tmp)
							taskFound = true
							break
						}
					}
				}

				// error: not able to find a node/task with the given groupByValue.
				if taskFound == false {
					err := fmt.Errorf("not able to find a node in a grpByValue list %s used in channel %s for role %s", valueList, channel.Name, currRole)
					zap.S().Errorf("TAG expansion failed due to incorrect groupByValue. %v", err)
					return err
				}
			}
			interConnect(tmpTasks)
		}
	}
	return nil
}

// exploreNewPeerRoleChannel takes in current and peer Role for the given channel. It then creates tasks/nodes for the peerRole based on the groupByValues associated with the channel.
// Since this peerRole is visited for the first time, tasks/nodes related to it will be created based on the channel configuration.
// Later during the exploreRole method other channels associated with this peerRole are explored.
func (t *tagExpander) exploreNewPeerRoleChannel(currRole string, peerRole string, channel openapi.Channel) error {
	zap.S().Debugf("Visiting peer role for the first time. Role %s connected via %s to role %s with grpValue: %v", peerRole, channel.Name, currRole, channel.GroupBy.Value)

	if containsInNestedArray(channel.GroupBy.Value, util.DefaultRealm) {
		// Default is a reserved keyword. Other groupBy values will be ignored if they are present alongside Default
		// While building a label for the peerRole nodes, system will combine the label of all the nodes that are converging to this node. This gives us a clear view of data association at every node.
		grpByValue := make([]string, 0)
		for _, tmp := range t.TopologyMap[currRole].Tasks {
			grpByValue = append(grpByValue, tmp.Label...)
		}

		tmpTask := t.buildWorker(peerRole, grpByValue, openapi.SYSTEM, 0)

		//update connectedLabel for nodes in currentRole and peerRole
		peerConnectedTasks := make([]string, 0)
		for _, tmp := range t.TopologyMap[currRole].Tasks {
			zap.S().Debugf("associating %s peerRole taskId (%s) to currRole node taskId (%s)", peerRole, tmpTask.TaskId, tmp.TaskId)
			tmp.ConnectedTaskIds[peerRole] = []string{tmpTask.TaskId}
			peerConnectedTasks = append(peerConnectedTasks, tmp.TaskId)
		}
		tmpTask.ConnectedTaskIds[currRole] = peerConnectedTasks
		t.TopologyMap[peerRole].Tasks = append(t.TopologyMap[peerRole].Tasks, tmpTask)
		zap.S().Debugf("peerRole taskId: %s with label:%s and connectedTasks:%s", tmpTask.TaskId, tmpTask.Label, tmpTask.ConnectedTaskIds)
	} else {
		// validate the labels of current role tasks/nodes are inline with the groupValues for this channel.
		// Error Case: for a given value in grpValue, if there exists no label associated with any node in the current-role it will result in an error.
		// Ignore Case: a label in the current role gets ignored because it was not mentioned in the groupByValues a warning will be printed.

		// check: grpByValues are valid
		// If the grpByValue is not found as a label in any node associated with the currRole => incorrect groupByValues
		var currRoleLabels []string
		for _, node := range t.TopologyMap[currRole].Tasks {
			currRoleLabels = append(currRoleLabels, node.Label...)
		}
		for _, valueArr := range channel.GroupBy.Value {
			for _, value := range valueArr {
				if !util.Contains(currRoleLabels, value) {
					err := fmt.Errorf("no node in current role associasted with the label %s for channel %s between (current) %s and %s (peer)", value, channel.Name, currRole, peerRole)
					zap.S().Errorf("TAG expansion failed due to incorrect groupByValue. %v", err)
					return err
				}
			}
		}

		// create peer nodes based on grpByValues
		tmpTasks := make([]*objects.Task, 0)
		for i, tmpLabel := range channel.GroupBy.Value {
			tmpTask := t.buildWorker(peerRole, tmpLabel, openapi.SYSTEM, i)

			// update connectedLabel
			peerConnectedTasks := make([]string, 0)
			for _, currNode := range t.TopologyMap[currRole].Tasks {
				// iterate the labels to determine which currRole node is connected to this peer
				for _, currNodeLabel := range currNode.Label {
					if util.Contains(tmpLabel, currNodeLabel) {
						zap.S().Debugf("peerRole node label:%s found in currRole taskId:%s", tmpLabel, currNode.TaskId)
						currNode.ConnectedTaskIds[peerRole] = append(currNode.ConnectedTaskIds[peerRole], tmpTask.TaskId)
						peerConnectedTasks = append(peerConnectedTasks, currNode.TaskId)
						break //found the node, move to next one
					}
				}
			}
			tmpTask.ConnectedTaskIds[currRole] = peerConnectedTasks

			// done with mapping one grpByValue, add it to the peerRole tmp list. Move one to next to create next such group
			tmpTasks = append(tmpTasks, tmpTask)
		}
		t.TopologyMap[peerRole].Tasks = tmpTasks
	}

	t.TopologyMap[peerRole].IsVisited = true
	return nil
}

// exploreVisitedPeerRoleChannel is called by exploreRole method when the given peer role was visited as part of another role exploration.
// As a result tasks there already exists tasks associated with this role. The main role of this function is to validate if channel adheres to the groupBy rules and finally update the connected task information.
func (t *tagExpander) exploreVisitedPeerRoleChannel(currRole string, peerRole string, channel openapi.Channel) error {
	zap.S().Debugf("peer role %s already visit. Checking current channel %s configuration before updating connected tasks ids", peerRole, channel.Name)

	if containsInNestedArray(channel.GroupBy.Value, util.DefaultRealm) {
		// if grpByValue is Default ==> either currRole or peerRole # of nodes count needs to be 1.
		// Default implies convergence to single node. If both the roles have multiple nodes. This is not feasible.
		peerNodesCount := len(t.TopologyMap[peerRole].Tasks)
		currNodesCount := len(t.TopologyMap[currRole].Tasks)
		zap.S().Debugf("currRole %s # of tasks (%d) while peerRole %s # of tasks: %d", currRole, currNodesCount, peerRole, peerNodesCount)
		if peerNodesCount != 1 && currNodesCount != 1 {
			err := fmt.Errorf("TAG expansion voilation. Inconsistent groupBy configuration while expanding %s connected to %s. The channel %s uses default value while the connected nodes have more than one instances created", peerRole, currRole, channel.Name)
			zap.S().Errorf("TAG expansion failed due to incorrect groupByValue. %v", err)
			return err
		}

		updateConnectedTasks := func(singleNodeRole string, multipleNodeRole string) {
			extLabel := make([]string, 0)
			tmpConnectedTasks := make([]string, 0)
			for _, tmp := range t.TopologyMap[multipleNodeRole].Tasks {
				tmp.ConnectedTaskIds[singleNodeRole] = append(tmp.ConnectedTaskIds[singleNodeRole], t.TopologyMap[singleNodeRole].Tasks[0].TaskId)
				fmt.Printf("adding back connection to multipleNodeRole: %s node | connectedTasks: %s\n", tmp.Label, tmp.ConnectedTaskIds)
				tmpConnectedTasks = append(tmpConnectedTasks, tmp.TaskId)
				extLabel = append(extLabel, tmp.Label...)
			}
			t.TopologyMap[singleNodeRole].Tasks[0].ConnectedTaskIds[multipleNodeRole] = append(t.TopologyMap[singleNodeRole].Tasks[0].ConnectedTaskIds[multipleNodeRole], tmpConnectedTasks...)
			fmt.Printf("currentLabel %s | extLabel : %s\n", t.TopologyMap[singleNodeRole].Tasks[0].Label, extLabel)
			t.TopologyMap[singleNodeRole].Tasks[0].Label = append(t.TopologyMap[singleNodeRole].Tasks[0].Label, extLabel...)
			fmt.Printf("singlNodeRole: %s | updated to label:%s | connectedTasks[%s]:%s\n", singleNodeRole, t.TopologyMap[singleNodeRole].Tasks[0].Label, multipleNodeRole, t.TopologyMap[singleNodeRole].Tasks[0].ConnectedTaskIds[multipleNodeRole])
		}

		// check peerRole first because it is not yet explored completely
		if peerNodesCount == 1 {
			updateConnectedTasks(peerRole, currRole)
		} else if currNodesCount == 1 {
			updateConnectedTasks(currRole, peerRole)
		}
	} else {
		// find the nodes in currRole and peerRole that are required to be grouped.
		// if not able to find any task/node in currRole, results in an error
		// while peerRole node finding there are few ways to handle this:
		// Method1 - stop expansion and return error. However, for uneven expanded topology such as testcase_hfl_uneven_depth this will not work.
		// Method2 (used) - if there is only 1 peerRole node, extend its label to include the missing label. However, if there are multiple peerRole nodes will result in error!
		// We extend the label for the peerRole node and not for the currRole node because peerRole node expansion has not been done, thus it can still be manipulated.
		zap.S().Debugf("channel %s between %s and %s groupByValue is %v", channel.Name, currRole, peerRole, channel.GroupBy.Value)
		for _, key := range channel.GroupBy.Value {
			// find the nodes in currRole that match the key
			tmpTasksCurrRole := make([]*objects.Task, 0)
			for _, tmp := range t.TopologyMap[currRole].Tasks {
				zap.S().Debugf("currRole taskId %s with label %s and looking for task with partial label %s", tmp.TaskId, tmp.Label, key)
				if partialMatchInNestedArray(tmp.Label, key) {
					zap.S().Debugf("taskId %s matches", tmp.TaskId)
					tmpTasksCurrRole = append(tmpTasksCurrRole, tmp)
				}
			}
			if len(tmpTasksCurrRole) == 0 {
				err := fmt.Errorf("not able to find label associated with %s in role %s", key, currRole)
				zap.S().Errorf("TAG expansion failed due to incorrect groupByValue. %v", err)
				return err
			}

			// find nodes in peerRole that match the groubBy label key
			tmpTasksPeerRole := make([]*objects.Task, 0)
			for _, tmp := range t.TopologyMap[peerRole].Tasks {
				zap.S().Debugf("peerRole taskId %s with label %s and looking for task with partial label %s", tmp.TaskId, tmp.Label, key)
				if partialMatchInNestedArray(tmp.Label, key) {
					zap.S().Debugf("taskId %s matches", tmp.TaskId)
					tmpTasksPeerRole = append(tmpTasksPeerRole, tmp)
				}
			}

			// if no node is found in peerRole
			if len(tmpTasksPeerRole) == 0 {
				zap.S().Debugf("not able to find label associated with %s in role %s. Next step check and extend label of existing peerRole node if valid", key, peerRole)

				if len(t.TopologyMap[peerRole].Tasks) == 1 {
					// update the label to include this key
					zap.S().Debugf("extending the label for %s taskId %s from %s", peerRole, t.TopologyMap[peerRole].Tasks[0].TaskId, t.TopologyMap[peerRole].Tasks[0].Label)
					t.TopologyMap[peerRole].Tasks[0].Label = append(t.TopologyMap[peerRole].Tasks[0].Label, key...)
					tmpTasksPeerRole = append(tmpTasksPeerRole, t.TopologyMap[peerRole].Tasks[0])
					zap.S().Debugf("updated label for %s taskId: %s to %s", peerRole, t.TopologyMap[peerRole].Tasks[0].TaskId, t.TopologyMap[peerRole].Tasks[0].Label)
				} else {
					// multiple nodes detected, too ambiguous for the system to decide which node to pick
					err := fmt.Errorf("not able to find label %s associated with nodes in role %s", key, currRole)
					zap.S().Errorf("TAG expansion failed due to incorrect groupByValue. %v", err)
					return err
				}
			}

			// associate connectedTaskIds
			for _, tmp1 := range tmpTasksCurrRole {
				for _, tmp2 := range tmpTasksPeerRole {
					tmp1.ConnectedTaskIds[peerRole] = append(tmp1.ConnectedTaskIds[peerRole], tmp2.TaskId)
					tmp2.ConnectedTaskIds[currRole] = append(tmp2.ConnectedTaskIds[currRole], tmp1.TaskId)
				}
			}
		}
	}
	return nil
}

func (t *tagExpander) buildWorker(roleName string, label []string, workerType openapi.TaskType, idx int) *objects.Task {
	role := t.RoleInfo[roleName].Role
	workerTask := t.getRoleTask(role)
	workerTask.Label = label
	// TODO 1. associate realm or compute information
	// 		2. fill the dataset info if role is DataConsumer
	workerTask.Configure(workerType, util.RandString(taskKeyLen), "", emptyDatasetUrl, idx)
	zap.S().Debugf("task created: %s node.Id:%s | label: %s", roleName, workerTask.TaskId, workerTask.Label)
	return workerTask
}

func (t *tagExpander) buildDataWorkers(role string) error {
	rInfo := t.RoleInfo[role]

	// if it is not a dataConsumer, or it is already visited, implies trainer nodes were already created
	if !rInfo.Role.IsDataConsumer || t.TopologyMap[rInfo.Role.Name].IsVisited {
		return nil
	}

	// getAssociatedDataset returns dataSpec associated with the given dataConsumer role
	getAssociatedDataset := func(dataRole string) (map[string][]string, error) {
		for roleNameKey, value := range t.JobBuilder.jobSpec.DataSpec.FromSystem {
			if roleNameKey == dataRole {
				return value, nil
			}
		}

		err := fmt.Errorf("no dataset associated with the data consumer %s", dataRole)
		return nil, err
	}

	//get associated data to the consumer
	dataSpec, err := getAssociatedDataset(rInfo.Role.Name)
	if err != nil {
		zap.S().Errorf("TAG expansion failed to located dataset. %v", err)
		return err
	}

	zap.S().Debugf("dataSpec for given jobId %s | %v\n", t.JobBuilder.jobSpec.Id, dataSpec)
	//TODO add FromUser handling

	// iterate dataset groups provided under dataSpec and for each datasetId create a task and attached the group label to it.
	tmpTasks := make([]*objects.Task, 0)
	for grp, datasets := range dataSpec {
		for i, _ := range datasets {
			task := t.buildWorker(rInfo.Role.Name, []string{grp}, openapi.SYSTEM, i)
			tmpTasks = append(tmpTasks, task)
		}
	}
	t.TopologyMap[rInfo.Role.Name].Tasks = tmpTasks
	t.TopologyMap[rInfo.Role.Name].IsVisited = true
	return nil
}

// //////////////////////////////////////////////////////////////////////////////
// Helper Functions
// //////////////////////////////////////////////////////////////////////////////

// getRoleInfo iterate through each role mentioned in the schema to create a task template.
// It includes attaching the zipped code associated with the role and updating the channel backend information if empty.
func (t *tagExpander) getRoleInfo() {
	dataRoles := make([]string, 0)
	templates := make(map[string]roleInfo)

	for _, role := range t.JobBuilder.schema.Roles {
		if role.IsDataConsumer {
			dataRoles = append(dataRoles, role.Name)
		}

		roleChannels := make([]openapi.Channel, 0)
		for _, channel := range t.JobBuilder.schema.Channels {
			if util.Contains(channel.Pair, role.Name) {
				roleChannels = append(roleChannels, channel)
			}
		}

		currRoleInfo := roleInfo{
			Role:     role,
			Channels: roleChannels,
		}
		templates[role.Name] = currRoleInfo
	}

	t.DataConsumerRoles = dataRoles
	t.RoleInfo = templates
}

// getRoleTask creates a task object for given role which is used by buildWorker to create the task
func (t *tagExpander) getRoleTask(role openapi.Role) *objects.Task {
	template := objects.Task{}
	JobConfig := &template.JobConfig
	JobConfig.Configure(t.JobBuilder.jobSpec, t.JobBuilder.jobParams.Brokers, t.JobBuilder.jobParams.Registry, role, t.JobBuilder.schema.Channels)

	template.JobId = JobConfig.Job.Id
	template.Role = role.Name
	template.IsDataConsumer = role.IsDataConsumer

	for i := range JobConfig.Channels {
		// check for channel's backend. If not present update it with default value
		if len(JobConfig.Channels[i].Backend) < 0 {
			JobConfig.Channels[i].Backend = t.JobBuilder.schema.DefaultBackend
		}

		// check channels and set default group if channel doesn't have groupBy attributes set
		if len(JobConfig.Channels[i].GroupBy.Value) < 0 {
			JobConfig.Channels[i].GroupBy.Type = groupByTypeTag
			JobConfig.Channels[i].GroupBy.Value = [][]string{{0: defaultGroup}}
		}
	}

	template.ZippedCode = t.JobBuilder.roleCode[role.Name]
	template.Label = make([]string, 0)
	template.ConnectedTaskIds = make(map[string][]string)
	return &template
}

// getPeerInfo find the roleInfo for the peer role connected to the given current role through given channel
func getPeerInfo(currRole string, channel openapi.Channel, schemaRoleInfo map[string]roleInfo) (roleInfo, error) {
	if !util.Contains(channel.Pair, currRole) {
		err := fmt.Errorf("role %s ot present in the channel %s", currRole, channel.Name)
		return roleInfo{}, err
	}

	peer := channel.Pair[0]
	if currRole == peer {
		peer = channel.Pair[1]
	}

	peerTmpl, ok := schemaRoleInfo[peer]
	if !ok {
		err := fmt.Errorf("no template found for peer role %s", peer)
		return roleInfo{}, err
	}

	return peerTmpl, nil
}

func partialMatchInNestedArray(input []string, keys []string) bool {
	for _, key := range keys {
		for _, value := range input {
			if reflect.DeepEqual(key, value) {
				return true
			}
		}
	}
	return false
}

func containsInNestedArray(grpValue [][]string, value string) bool {
	for _, valArray := range grpValue {
		for _, v := range valArray {
			if v == value {
				return true
			}
		}
	}
	return false
}

// printTopology prints the expanded TAG nodes in log
func (t *tagExpander) printTopology() {
	zap.S().Debugf("printing expanded topology for jobId %s with schema version %s", t.JobBuilder.jobSpec.Id, t.JobBuilder.jobSpec.SchemaVersion)
	connectedRoleQ := make([]string, 0)
	isVisited := make(map[string]bool)

	printDetails := func(role string) {
		zap.S().Debugf("role: %s contains #tasks %d", t.TopologyMap[role].Role, len(t.TopologyMap[role].Tasks))
		for i, node := range t.TopologyMap[role].Tasks {
			zap.S().Debugf("%d | %v", i, node.ToString())
		}
	}

	connectedRoles := func(currRole string) {
		currRoleInfo := t.RoleInfo[currRole]
		for _, channel := range currRoleInfo.Channels {
			peerRoleInfo, _ := getPeerInfo(currRole, channel, t.RoleInfo)
			_, isDone := isVisited[peerRoleInfo.Role.Name]

			//peerRole already visited or is already in the Q don't add again
			if isDone || util.Contains(connectedRoleQ, peerRoleInfo.Role.Name) {
				continue
			}
			connectedRoleQ = append(connectedRoleQ, peerRoleInfo.Role.Name)
		}
	}

	//print the dataRoles and add their connected role into Q
	for _, role := range t.DataConsumerRoles {
		printDetails(role)
		isVisited[role] = true
		connectedRoles(role)
	}

	//Dequeue roles and keep Queueing the roles connected to the dequeued role.
	Qsize := len(connectedRoleQ)
	for Qsize > 0 {
		//dequeue
		role := connectedRoleQ[0]
		isVisited[role] = true
		connectedRoleQ = connectedRoleQ[1:]

		printDetails(role)
		connectedRoles(role)
		Qsize = len(connectedRoleQ)
	}
}
