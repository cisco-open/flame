#!/usr/bin/env bash

# This script requires openapi-generator.
# 
# There are a couple of ways to install openapi-generator.
# For example, it can be installed by using HomeBrew in mac os.
# i.e., brew install openapi-generator
#
# For more details for installing openapi-generator, refer to
# https://openapi-generator.tech/docs/installation/.

source openapi-yaml.sh

generate_openapi_yaml

echo "Generating Specification for configuration file :" $conf_file

conf_path=${conf_dir}/${conf_file}
openapi-generator generate -i ${conf_path} -g go-server \
		  --additional-properties=sourceFolder=openapi

rm -rf  .openapi-generator .openapi-generator-ignore  main.go Dockerfile README.md go.mod api

rm -f ${conf_path}
