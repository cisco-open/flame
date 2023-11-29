/**
 * Copyright 2023 Cisco Systems, Inc. and its affiliates
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

export interface Job {
    createdAt: string;
    endedAt: string;
    id: string;
    startedAt: string;
    state: string;
    updatedAt: string;
    name: string;
    experimentId?: string;
}

export interface DatasetPayload {
    role: string,
    datasetGroups: {
        [key:string]: string[]
    }
}

export interface DatasetControls {
    label: string,
    controls: string[]
}

