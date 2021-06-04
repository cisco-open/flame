#!/usr/bin/env bash

# install npm
# MAC OS: brew install node

npx --yes redoc-cli bundle -o ../docs/index.html ../api/openapi.yaml
