all: build 
.PHONY: build push clean dep fmt local test

user := $(shell whoami)
registry := containers.cisco.com/research

project := fledge
now := $(shell date +"%Y%m%d%H%M%S")
uri := ${project}:${user}-${now}

dep:
	@go mod tidy

fmt:
	@go fmt ./...

build:
	@docker build -f build/Dockerfile.dev --target golang-build --tag ${uri} .

push: build
	@docker image tag ${uri} ${registry}/${uri}
	@docker image push ${registry}/${uri}

clean:
	@docker image prune --force

local: dep
	@mkdir -p ./build/bin
	@go build -o ./build/bin ./...

install: local
	mkdir -p $(HOME)/.fledge/bin
	cp -r ./build/bin $(HOME)/.fledge

test: dep
	@go test -v ./...
