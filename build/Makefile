# Copyright 2022 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

all: build 
.PHONY: build push clean dep fmt local test

user := $(shell whoami)
registry := containers.cisco.com/research

project := flame
now := $(shell date +"%Y%m%d%H%M%S")
uri := ${project}:${user}-${now}

dep:
	@go mod tidy

fmt:
	@go fmt ./...

build:
	@docker build -f Dockerfile --tag ${uri} .

push: build
	@docker image tag ${uri} ${registry}/${uri}
	@docker image push ${registry}/${uri}

clean:
	@docker image prune --force

local: dep
	@mkdir -p ./build/bin
	@go build -o ./build/bin ./...

install: local
	mkdir -p $(HOME)/.flame/bin
	cp -r ./build/bin $(HOME)/.flame

test: dep
	@go test -v ./...
