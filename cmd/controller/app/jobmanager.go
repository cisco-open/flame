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

package app

import (
	"fmt"

	"wwwin-github.cisco.com/eti/fledge/cmd/controller/app/jobeventq"
)

type JobManager struct {
	jobEventQ *jobeventq.EventQ
}

func NewJobManager(jobEventQ *jobeventq.EventQ) (*JobManager, error) {
	if jobEventQ == nil {
		return nil, fmt.Errorf("job event queue is nil")
	}

	jobManager := &JobManager{
		jobEventQ: jobEventQ,
	}

	return jobManager, nil
}

func (mgr *JobManager) Do() {
	for {
		event := mgr.jobEventQ.Dequeue()
		mgr.handle(event)
	}
}

func (mgr *JobManager) handle(event *jobeventq.JobEvent) {
	// TODO: implement me!
	event.ErrCh <- fmt.Errorf("not yet implemented")
}
