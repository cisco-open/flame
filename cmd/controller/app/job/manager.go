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
)

type Manager struct {
	jobEventQ *EventQ

	notifierEp string

	jobQueues map[string]*EventQ
	mutexQ    *sync.Mutex
}

func NewManager(jobEventQ *EventQ, notifierEp string) (*Manager, error) {
	if jobEventQ == nil {
		return nil, fmt.Errorf("job event queue is nil")
	}

	manager := &Manager{
		jobEventQ: jobEventQ,

		notifierEp: notifierEp,
		jobQueues:  make(map[string]*EventQ),
		mutexQ:     new(sync.Mutex),
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
			jobHandler := NewHandler(event.JobStatus.Id, eventQ, mgr.jobQueues, mgr.mutexQ, mgr.notifierEp)
			go jobHandler.Do()
		}
		eventQ.Enqueue(event)
		mgr.mutexQ.Unlock()
	}
}
