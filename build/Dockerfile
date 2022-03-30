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

FROM golang:1.16.4-buster AS golang-build
WORKDIR /src
ENV CGO_ENABLED=0
RUN --mount=target=. \
    --mount=type=cache,target=/root/.cache/go-build \
    go test -v ./...

RUN --mount=target=. \
    --mount=type=cache,target=/root/.cache/go-build \
    go build -o /flame/ ./...

RUN mkdir -p /tmp/job-template
COPY templates/job/ /tmp/job-template/

FROM python:3.9.7-buster AS python-build
WORKDIR /src
COPY ./lib/python /src
RUN python3 setup.py bdist_wheel && cp dist/flame-*.whl /tmp/

FROM myungjinlee/mlpack:cpu
COPY --from=golang-build /flame/* /usr/bin/

RUN mkdir -p /flame/deployment
COPY --from=golang-build /tmp/job-template/ /flame/deployment/

# Install flame python library
COPY --from=python-build /tmp/*.whl /tmp/
RUN pip3 install /tmp/*.whl && rm -f /tmp/*.whl
# this python script configures python packages
RUN flame-config
