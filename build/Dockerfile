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

FROM golang:1.22.0 AS golang-build
WORKDIR /src
ENV CGO_ENABLED=0

RUN --mount=target=. \
    --mount=type=cache,target=/root/.cache/go-build \
    go build -o /flame/ ./...

FROM python:3.9.7-buster AS python-build
WORKDIR /src
COPY ./lib/python /src
RUN python3 setup.py bdist_wheel && cp dist/flame-*.whl /tmp/

# available image tags: pytorch-cpu, tensorflow-cpu, allinone-cpu
#                       pytorch-cuda11.3, tensorflow-cuda11.3, allinone-cuda11.3
# note: allinone-cuda11.3 is the largest image
FROM myungjinlee/mlpack:allinone-cuda11.3

# Install required dependencies and Node.js
ENV NODE_VERSION=17
RUN apt-get update && \
    apt-get install -y curl && \
    curl -sL https://deb.nodesource.com/setup_${NODE_VERSION}.x | bash - && \
    apt-get install -y nodejs

COPY --from=golang-build /flame/* /usr/bin/

# Install flame python library
COPY --from=python-build /tmp/*.whl /tmp/


COPY ./dashboard /dashboard_src
WORKDIR /dashboard_src
RUN npm install -g serve
RUN npm install
RUN npm run build
RUN cp -a /dashboard_src/build/. /dashboard
RUN rm -rf /dashboard_src
WORKDIR /

RUN pip3 install /tmp/*.whl && rm -f /tmp/*.whl
# this python script configures python packages
RUN flame-config
