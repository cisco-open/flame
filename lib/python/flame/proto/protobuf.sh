#!/usr/bin/env bash

protoc -I=./ --python_out=./ ./registry_msg.proto
protoc -I=./ --python_out=./ ./backend_msg.proto
