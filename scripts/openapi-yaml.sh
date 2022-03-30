#!/usr/bin/env bash
# Copyright 2022 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0


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
