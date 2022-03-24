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

package database

import (
	"fmt"
	"strings"

	"github.com/cisco-open/flame/cmd/controller/app/database/mongodb"
	"github.com/cisco-open/flame/pkg/util"
)

func NewDBService(uri string) (DBService, error) {
	dbName := strings.Split(uri, ":")[0]

	var err error

	switch dbName {
	case util.MONGODB:
		return mongodb.NewMongoService(uri)

	case util.MySQL:
		fallthrough

	default:
		err = fmt.Errorf("unknown DB type: %s", dbName)
	}

	return nil, err
}
