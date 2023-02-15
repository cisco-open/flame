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
		  --additional-properties=sourceFolder=openapi,outputAsLibrary=true

./addlicense.sh

go fmt ./...

rm -rf  .openapi-generator .openapi-generator-ignore README.md api

rm -f ${conf_path}
