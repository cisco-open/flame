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

export interface GetRunsPayload {
    experiment_ids: string[];
    max_results: number;
    order_by: string[];
    run_view_type: RunViewType;
}

export enum RunViewType {
    activeOnly = 'ACTIVE_ONLY',
    deletedOnly = 'DELETED_ONLY',
}

export interface RunResponse {
    runs: Run[];
}

export interface Run {
    data: RunData;
    info: RunInfo;
    startDate: string;
    endDate: string;
    taskId: string;
}

export interface RunData {
    metrics: Metric[];
    parameters: Parameter[];
    tags: Tag[];
}

export interface RunInfo {
    artifact_uri: string;
    end_time: number;
    experiment_id: string;
    lifecycle_stage: string;
    run_id: string;
    run_name: string;
    run_uuid: string;
    start_time: number;
    status: string;
    user_id: string;
}

export interface Metric {
    key: string;
    value: number;
    timestamp: number;
    step: number;
}

export interface Parameter {
    key: string;
    value: string;
}

export interface Tag {
    key: string;
    value: string;
}