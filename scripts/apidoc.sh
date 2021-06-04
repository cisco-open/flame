#!/usr/bin/env bash

# install npm
# MAC OS: brew install node

npx --yes redoc-cli bundle -o ../docs/openapi.html ../api/openapi.yaml
