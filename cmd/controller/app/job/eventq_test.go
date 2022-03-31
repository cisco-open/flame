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
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/cisco-open/flame/pkg/openapi"
)

func TestNewJobEvent(t *testing.T) {
	jobStatus := openapi.JobStatus{}
	jobEvent := NewJobEvent("some_user", jobStatus)
	assert.NotNil(t, jobEvent)
	assert.NotNil(t, jobEvent.ErrCh)
}

func TestNewEventQ(t *testing.T) {
	eventQ := NewEventQ(0)
	assert.NotNil(t, eventQ)
	assert.Equal(t, defaultQueueSize, cap(eventQ.buf))

	eventQ = NewEventQ(1)
	assert.NotNil(t, eventQ)
	assert.Equal(t, 1, cap(eventQ.buf))
}

func TestEnqueue(t *testing.T) {
	eventQ := NewEventQ(1)
	assert.NotNil(t, eventQ)

	jobEvent := &JobEvent{}
	err := eventQ.Enqueue(jobEvent)
	assert.NotNil(t, err)
	assert.Equal(t, "error channel in jobEvent not initialized", err.Error())

	jobStatus := openapi.JobStatus{}
	jobEvent = NewJobEvent("some_user", jobStatus)
	err = eventQ.Enqueue(jobEvent)
	assert.Nil(t, err)

	err = eventQ.Enqueue(jobEvent)
	assert.NotNil(t, err)
	assert.Equal(t, "event queue is full; discarding the event", err.Error())
}

func TestDequeue(t *testing.T) {
	eventQ := NewEventQ(1)
	assert.NotNil(t, eventQ)

	jobStatus := openapi.JobStatus{Id: "12345"}
	jobEvent := NewJobEvent("some_user", jobStatus)
	err := eventQ.Enqueue(jobEvent)
	assert.Nil(t, err)

	dequeuedEvent := eventQ.Dequeue()
	assert.NotNil(t, dequeuedEvent)
	assert.Equal(t, jobEvent.JobStatus.Id, dequeuedEvent.JobStatus.Id)
}
