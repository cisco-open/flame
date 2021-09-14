#!/usr/bin/env bash

go install github.com/google/addlicense@v1.0.0

# add license for go files
find .. -type f -name *.go -exec ${HOME}/go/bin/addlicense -f ../LICENSE '{}' +

# add license for python files
find .. -type f -name *.go -exec ${HOME}/go/bin/addlicense -f ../LICENSE '{}' +
