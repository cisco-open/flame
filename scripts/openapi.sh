#!/usr/bin/env bash

source openapi-yaml.sh

generate_openapi_yaml

echo "Generating Specification for configuration file :" $conf_file

conf_path=${conf_dir}/${conf_file}
openapi-generator generate -i ${conf_path} -g go-server \
		  --additional-properties=sourceFolder=openapi

rm -rf  .openapi-generator .openapi-generator-ignore  main.go Dockerfile README.md go.mod api

rm -f ${conf_path}
