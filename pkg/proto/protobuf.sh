#!/usr/bin/env bash

clear
#protoc --go_out=. --go_opt=paths=source_relative --go-grpc_out=. --go-grpc_opt=paths=source_relative agent.proto
protoc --go_out=./go --go_opt=paths=source_relative --go-grpc_out=./go --go-grpc_opt=paths=source_relative controller.proto

#mv agent.pb.go ../../cmd/agent/grpc
#mv controller*.pb.go ../../cmd/controller/grpc/internal
