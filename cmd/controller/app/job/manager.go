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
	"fmt"
	"sync"

	"github.com/cisco-open/flame/cmd/controller/app/database"
	"github.com/cisco-open/flame/cmd/controller/config"
	"go.uber.org/zap"
)

type Manager struct {
	dbService database.DBService
	jobEventQ *EventQ

	notifier  string
	jobParams config.JobParams
	platform  string

	jobQueues map[string]*EventQ
	mutexQ    *sync.Mutex
}

func NewManager(dbService database.DBService, jobEventQ *EventQ, notifier string, jobParams config.JobParams,
	platform string) (*Manager, error) {
	if jobEventQ == nil {
		return nil, fmt.Errorf("job event queue is nil")
	}

	manager := &Manager{
		dbService: dbService,
		jobEventQ: jobEventQ,

		jobParams: jobParams,
		notifier:  notifier,
		platform:  platform,
		jobQueues: make(map[string]*EventQ),
		mutexQ:    new(sync.Mutex),
	}

	return manager, nil
}

func (mgr *Manager) Do() {
	for {
		event := mgr.jobEventQ.Dequeue()

		mgr.mutexQ.Lock()
		eventQ, ok := mgr.jobQueues[event.JobStatus.Id]
		if !ok {
			eventQ = NewEventQ(0)
			mgr.jobQueues[event.JobStatus.Id] = eventQ
			jobHandler, err := NewHandler(mgr.dbService, event.JobStatus.Id, eventQ, mgr.jobQueues, mgr.mutexQ,
				mgr.notifier, mgr.jobParams, mgr.platform)
			if err != nil {
				mgr.mutexQ.Unlock()
				zap.S().Warnf("Failed to create a job handler: %v", err)
				continue
			}
			go jobHandler.Do()
		}
		eventQ.Enqueue(event)
		mgr.mutexQ.Unlock()
	}
}
