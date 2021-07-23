#!/usr/bin/env bash

conf_root_folder="../api/"
conf_file="openapi.yaml"

if [ $# -eq 0 ]
  then
    echo "No arguments supplied. Using default conf file : " $conf_file
  else
    conf_file=$1
fi

echo "Generating Specification for configuration file :" $conf_file

openapi-generator generate -i $conf_root_folder"/"$conf_file -g go-server \
		  --additional-properties=sourceFolder=openapi

rm -rf  .openapi-generator .openapi-generator-ignore  main.go Dockerfile README.md go.mod api
