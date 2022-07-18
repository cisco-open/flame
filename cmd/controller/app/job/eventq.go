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

	"github.com/cisco-open/flame/pkg/openapi"
)

const (
	defaultQueueSize = 100
)

type JobEvent struct {
	Requester string
	JobStatus openapi.JobStatus
	ErrCh     chan error
}

type EventQ struct {
	buf chan *JobEvent
}

func NewJobEvent(requester string, jobStatus openapi.JobStatus) *JobEvent {
	jobEvent := &JobEvent{
		Requester: requester,
		JobStatus: jobStatus,
		ErrCh:     make(chan error, 1),
	}

	return jobEvent
}

func NewEventQ(qSize int) *EventQ {
	if qSize <= 0 {
		qSize = defaultQueueSize
	}

	q := &EventQ{
		buf: make(chan *JobEvent, qSize),
	}

	return q
}

func (q *EventQ) Enqueue(jobEvent *JobEvent) error {
	if jobEvent.ErrCh == nil {
		return fmt.Errorf("error channel in jobEvent not initialized")
	}

	select {
	case q.buf <- jobEvent: // put jobEvent in the channel unless it's full

	default:
		return fmt.Errorf("event queue is full; discarding the event")
	}

	return nil
}

func (q *EventQ) Dequeue() *JobEvent {
	return <-q.buf
}

func (q *EventQ) GetJobEventBuffer() <-chan *JobEvent {
	return q.buf
}
