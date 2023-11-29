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

BASE_FILE=license_header.txt
TMP_FILE=header.tmp

# update year
year=$(date +'%Y')
cat ${BASE_FILE} | sed -e "s/2022/${year}/" > ${TMP_FILE}

pushd ..

FILE_EXTS=(go html js proto py ts tsx yaml yml)
for ext in ${FILE_EXTS[*]}; do
    find . -not -path '*/.*' -type f -name *.${ext} \
	 -exec ${HOME}/go/bin/addlicense -f ./scripts/${TMP_FILE} '{}' +
done

FILES=(Dockerfile)
for file in ${FILES[*]}; do
    find . -not -path '*/.*' -type f -name ${file} \
	 -exec ${HOME}/go/bin/addlicense -f ./scripts/${TMP_FILE} '{}' +
done

popd

rm -f ${TMP_FILE}
