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

export interface MetricsRequestParams {
  run_uuid: string;
  metric_key: string;
}

export interface ArtifactsRequestParams {
  run_uuid: string;
  path?: string;
}


export interface MetricsTimelineDataItem {
  start: number;
  category: string;
  key: string;
  step: number;
  value: number;
  timestamp: number;
}

export interface MetricsTimelineData {
  [key: string]: MetricsTimelineDataItem[]
}

export interface TimelineGroup {
  id: string;
  title: JSX.Element;
  name: string;
}

export interface Timelineitem {
  id: string;
  group: string;
  title: string;
  start: number;
  end: number;
  bgColor: string;
}