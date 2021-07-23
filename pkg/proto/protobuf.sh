#!/usr/bin/env bash

clear

#protoc --go_out=./go --go_opt=paths=source_relative --go-grpc_out=./go --go-grpc_opt=paths=source_relative notification.proto
protoc --go_out=./go/notification --go_opt=paths=source_relative --go-grpc_out=./go/notification --go-grpc_opt=paths=source_relative notification.proto
#protoc --go_out=./go/agent --go_opt=paths=source_relative --go-grpc_out=./go/agent --go-grpc_opt=paths=source_relative agent.proto

#protoc --python_out=./python/agent agent.proto
python3 -m grpc_tools.protoc -I./ --python_out=../../lib/python/fledge/proto/agent --grpc_python_out=../../lib/python/fledge/proto/agent ./agent.proto