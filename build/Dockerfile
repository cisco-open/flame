# FROM containers.cisco.com/eti-sre/sre-golang-docker:latest AS base
FROM golang:1.16.4-buster AS base

WORKDIR /src
ENV CGO_ENABLED=0

FROM base AS build
RUN --mount=target=. \
    --mount=type=cache,target=/root/.cache/go-build \
    go build -o /fledge/ ./...

FROM base AS test
RUN --mount=target=. \
    --mount=type=cache,target=/root/.cache/go-build \
    go test -v ./...
