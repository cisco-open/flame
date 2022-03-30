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


go install github.com/google/addlicense@v1.0.0

LICENSE_FILE=license.tmp

# update year
year=$(date +'%Y')
cat ../LICENSE | sed -e "s/2021/${year}/" > ${LICENSE_FILE}

# add license for go files
find .. -type f -name *.go -exec ${HOME}/go/bin/addlicense -f ${LICENSE_FILE} '{}' +

# add license for python files
find .. -type f -name *.py -exec ${HOME}/go/bin/addlicense -f ${LICENSE_FILE} '{}' +

rm -f ${LICENSE_FILE}
