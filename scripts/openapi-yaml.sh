#!/usr/bin/env bash

# This script requires a python implementation of the mustache templating language.
# https://github.com/noahmorrison/chevron
# 
# To install cheron: pip3 install chevron

conf_dir="../api/"
conf_file="openapi.yaml"

function generate_openapi_yaml {
    pushd ${conf_dir}
    chevron -d empty.json -p . -e yml ${conf_file}.mustache > ${conf_file}
    popd
}
