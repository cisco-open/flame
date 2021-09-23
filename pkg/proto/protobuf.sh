#!/usr/bin/env bash

go install google.golang.org/protobuf/cmd/protoc-gen-go
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@v1.1.0

export PATH=$PATH:$HOME/go/bin

case "$1" in
    "notification"|"agent")
	mkdir -p $1
	protoc --go_out=$1 \
	       --go_opt=paths=source_relative \
	       --go-grpc_out=$1 \
	       --go-grpc_opt=paths=source_relative \
	       $1.proto
	;;
    *)
	echo "usage: ./protobuf.sh [ notification | agent ]"
	exit 1
	;;
esac

# TODO: revisit the following later

#protoc --go_out=./go --go_opt=paths=source_relative --go-grpc_out=./go --go-grpc_opt=paths=source_relative notification.proto

#protoc --go_out=./go/fledgelet --go_opt=paths=source_relative --go-grpc_out=./go/fledgelet --go-grpc_opt=paths=source_relative fledgelet.proto

#protoc --python_out=./python/fledgelet fledgelet.proto
#python3 -m grpc_tools.protoc -I./ --python_out=../../lib/python/fledge/proto/agent --grpc_python_out=../../lib/python/fledge/proto/agent ./agent.proto
