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


apiserver_ip=$(kubectl get svc -n flame  | grep flame-apiserver | awk '{print $4}')

FLAME_DIR=$HOME/.flame

mkdir -p $FLAME_DIR

cat > $FLAME_DIR/config.yaml <<EOF
# local fiab env
---
apiserver:
  endpoint: http://$apiserver_ip:10100
user: foo
EOF
