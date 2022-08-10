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
	"sync"

	"github.com/cisco-open/flame/cmd/controller/app/database"
	"github.com/cisco-open/flame/cmd/controller/config"
)

const (
	defaultHandler = "default"
)

type handler interface {
	Do()
}

func NewHandler(handlerType string, dbService database.DBService, jobId string, eventQ *EventQ,
	jobQueues map[string]*EventQ, mu *sync.Mutex, notifier string, jobParams config.JobParams,
	bInsecure bool, bPlain bool) (handler, error) {
	var hdlr handler
	var err error

	switch handlerType {
	case defaultHandler:
		hdlr, err = NewDefaultHandler(dbService, jobId, eventQ, jobQueues, mu, notifier, jobParams, bInsecure, bPlain)

	default:
		err = fmt.Errorf("unknown handler type")
	}

	if err != nil {
		return nil, err
	}

	return hdlr, nil
}
