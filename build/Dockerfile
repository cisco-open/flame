FROM golang:1.16.4-buster AS base

WORKDIR /src
ENV CGO_ENABLED=0
ADD . .

FROM base AS build
RUN --mount=type=cache,target=/root/.cache/go-build \
    go build -o /fledge/ ./...

FROM base AS test
RUN --mount=type=cache,target=/root/.cache/go-build \
    go test -v ./...
