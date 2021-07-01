all: build 
.PHONY: build test push clean dep fmt local local_test

user := $(shell whoami)
registry := containers.cisco.com/research

project := fledge
now := $(shell date +"%Y%m%d%H%M%S")
uri := ${project}:${user}-${now}

dep:
	@go mod tidy

fmt:
	@go fmt ./...

build: test
	@docker build -f build/Dockerfile.dev --target build --tag ${uri} .

test:
	@docker build -f build/Dockerfile.dev --target test .

push: build
	@docker image tag ${uri} ${registry}/${uri}
	@docker image push ${registry}/${uri}

clean:
	@docker image prune --force

local: dep
	@mkdir -p ./build/bin
	@go build -o ./build/bin ./...

local_test: dep
	@go test -v ./...
