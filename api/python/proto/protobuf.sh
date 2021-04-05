#!/usr/bin/env bash

protoc -I=./ --python_out=./ ./registry_msg.proto
