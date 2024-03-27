/**
 * Copyright 2024 Cisco Systems, Inc. and its affiliates
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

export const JOB_STATE_COLOR: { [key: string]: { color: string, borderColor: string } } = {
  ready: {
    borderColor: 'rgba(75, 192, 192, 0.8)',
    color: 'rgba(75, 192, 192, 1)',
  },
  completed: {
    borderColor: 'rgba(54, 162, 235, 0.8)',
    color: 'rgba(54, 162, 235, 1)',
  },
  failed: {
    borderColor: 'rgba(255, 99, 132, 0.8)',
    color: 'rgba(255, 99, 132, 1)',
  },
  terminated: {
    borderColor: 'rgba(247, 2, 2, 0.8)',
    color: 'rgba(247, 2, 2, 1)',
  },
  starting: {
    borderColor: 'rgba(69, 237, 181, 0.8)',
    color: 'rgba(69, 237, 181, 1)',
  },
  applying: {
    borderColor: 'rgba(245, 226, 15, 0.8)',
    color: 'rgba(245, 226, 15, 1)',
  },
  deploying: {
    borderColor: 'rgba(190, 7, 227, 0.8)',
    color: 'rgba(190, 7, 227, 1)',
  },
  stopping: {
    borderColor: 'rgba(145, 6, 52, 0.8)',
    color: 'rgba(145, 6, 52, 1)',
  },
  running: {
    borderColor: 'rgba(245, 169, 5, 0.8)',
    color: 'rgba(245, 169, 5, 1)',
  },
}