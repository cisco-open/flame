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

export interface JobForm {
    design: string | undefined;
    hyperParameters: any | undefined;
    basemodelName: string | undefined;
    basemodelVersion: string | undefined;
    backend: string | undefined;
    maxRunTime: string | undefined;
    priority: string | undefined;
    datasets: string | undefined;
    dependencies: string | undefined;
    optimizerName: string | undefined;
    optimizerKwargs: string | undefined;
    selectorKwargs: string | undefined;
    selectorName: string | undefined;
    designId?: string | undefined;
    dataSpec: any;
}