#!/usr/bin/env bash

openapi-generator generate -i ../api/openapi.yaml -g go-server \
		  --additional-properties=sourceFolder=openapi

rm -rf  .openapi-generator api .openapi-generator-ignore Dockerfile README.md go.mod
