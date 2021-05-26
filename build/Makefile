all: build 
.PHONY: build test push clean fmt

user := $(shell whoami)
registry := containers.cisco.com/research

project := fledge
now := $(shell date +"%Y%m%d%H%M%S")
uri := ${project}:${user}-${now}


build: test
	@docker build -f build/Dockerfile --target build --tag ${uri} .

test:
	@docker build -f build/Dockerfile --target test .

push: build
	@docker image tag ${uri} ${registry}/${uri}
	@docker image push ${registry}/${uri}

clean:
	@docker image prune --force

fmt:
	@go fmt ./...
