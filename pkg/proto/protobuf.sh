#!/usr/bin/env bash
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


go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.28
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@v1.2

export PATH=$PATH:$HOME/go/bin

case "$1" in
    "agent"|"notification")
	mkdir -p $1
	protoc --go_out=$1 \
	       --go_opt=paths=source_relative \
	       --go-grpc_out=$1 \
	       --go-grpc_opt=paths=source_relative \
	       $1.proto
	;;
    "meta")
	mkdir -p $1
	protoc --go_out=$1 \
	       --go_opt=paths=source_relative \
	       --go-grpc_out=$1 \
	       --go-grpc_opt=paths=source_relative \
	       $1.proto
	# To run the following command, grpcio-tools is needed.
	# if it's not installed, run "pip install grpcio-tools"
	python -m grpc_tools.protoc -I=./ --python_out=$1 --grpc_python_out=$1 $1.proto
	;;
    *)
	echo "usage: ./protobuf.sh [ agent | meta | notification ]"
	exit 1
	;;
esac
