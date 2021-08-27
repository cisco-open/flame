#!/usr/bin/env bash

# install npm
# MAC OS: brew install node

source openapi-yaml.sh

generate_openapi_yaml

conf_path=${conf_dir}/${conf_file}

npx --yes redoc-cli bundle -o ../docs/index.html ${conf_path}

rm -f ${conf_path}
