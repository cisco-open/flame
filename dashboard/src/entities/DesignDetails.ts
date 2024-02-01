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

import { Design } from "./Design";

export interface DesignDetails extends Design {
    schema: Schema;
}

export interface Schema {
    channels: Channel[];
    name: string;
    description: string;
    roles: Role[];
}

export interface Channel {
    description: string;
    funcTags: { [key: string]: string[] };
    groupBy: GroupBy;
    name: string;
    pair: string[];
    index?: number;
}

export interface FuncTags {
    aggregator?: string[];
    trainer?: string[];
}

export interface GroupBy {
    type: string;
    value: string[];
}

export interface Role {
    description: string;
    groupAssociation: GroupAssociation[];
    isDataConsumer: boolean;
    name: string;
    index?: number;
    replica?: number;
    previousName?: string;
}

export interface GroupAssociation {
    [key: string]: string;
}

export interface MappedFuncTag { roleName: string, funcTags: {
    value: string,
    selected: boolean,
    disabled: boolean,
}[] };