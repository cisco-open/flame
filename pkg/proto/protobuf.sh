#!/usr/bin/env bash

clear

#protoc --go_out=./go --go_opt=paths=source_relative --go-grpc_out=./go --go-grpc_opt=paths=source_relative notification.proto
protoc --go_out=./go/notification --go_opt=paths=source_relative --go-grpc_out=./go/notification --go-grpc_opt=paths=source_relative notification.proto