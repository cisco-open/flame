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

import { FitViewOptions } from "reactflow";
import CustomNodeNoInteraction from "../../components/custom-node-no-interaction/CustomNodeNoInteraction";
import CustomNode from "../../components/custom-node/CustomNode";
import FloatingEdge from "../../components/floating-edge/FloatingEdge";
import InvisibleEdge from "../../components/invisible-edge/InvisibleEdge";
import PlaceholderNode from "../../components/placeholder-node/PlaceholderNode";
import SelfConnectingEdge from "../../components/self-connecting-edge/SelfConnectingEdge";
import NoInteractionEdge from "../../components/no-interaction-edge/NoInteractionEdge";
import SelfConnectingNoInteraction from "../../components/self-containing-edge-no-interaction/SelfConnectingNoInteraction";

export const initialNodes = [
  {
    id: '1',
    data: { label: 'Hello' },
    position: { x: 0, y: 0 },
  },
  {
    id: '2',
    data: { label: 'World' },
    position: { x: 100, y: 100 },
  },
];

export const fitViewOptions: FitViewOptions = {
  padding: 3,
  nodes: [],
  includeHiddenNodes: true,
}


export const connectionLineStyle = {
  strokeWidth: 3,
  stroke: 'black',
};

export const defaultEdgeOptions = {
  style: { strokeWidth: 3, stroke: 'black' },
  type: 'floating',
};

export const initialEdges = [{ id: '1-2', source: '1', target: '2', label: 'to the' }];

export const edgeTypes = {
  floating: FloatingEdge as unknown as any,
  selfConnecting: SelfConnectingEdge,
  invisible: InvisibleEdge as unknown as any,
  noInteraction: NoInteractionEdge as unknown as any,
  selfConnectingNoInteraction: SelfConnectingNoInteraction,
}

export const nodeTypes = {
  custom: CustomNode,
  customNodeNoInteraction: CustomNodeNoInteraction,
  placeholder: PlaceholderNode,
}

export const FUNC_TAGS = {
  TAG_FETCH: 'fetch',
  TAG_UPLOAD: 'upload',
  TAG_DISTRIBUTE: 'distribute',
  TAG_AGGREGATE: 'aggregate',
  TAG_COORDINATE_WITH_TOP_AGG: "coordinateWithTopAgg",
  TAG_COORDINATE_WITH_MID_AGG: "coordinateWithMidAgg",
  TAG_COORDINATE_WITH_TRAINER: "coordinateWithTrainer",
}

export const FUNC_TAGS_MAPPING = [
  {
    fileValue: 'flame.mode.horizontal.trainer',
    funcTags: [FUNC_TAGS.TAG_FETCH, FUNC_TAGS.TAG_UPLOAD]
  },
  {
    fileValue: 'flame.mode.horizontal.top_aggregator',
    funcTags: [FUNC_TAGS.TAG_DISTRIBUTE, FUNC_TAGS.TAG_AGGREGATE]
  },
  {
    fileValue: 'flame.mode.horizontal.middle_aggregator',
    funcTags: [FUNC_TAGS.TAG_DISTRIBUTE, FUNC_TAGS.TAG_AGGREGATE, FUNC_TAGS.TAG_FETCH, FUNC_TAGS.TAG_UPLOAD]
  },
  {
    fileValue: 'flame.mode.horizontal.syncfl.middle_aggregator',
    funcTags: [FUNC_TAGS.TAG_DISTRIBUTE, FUNC_TAGS.TAG_AGGREGATE, FUNC_TAGS.TAG_FETCH, FUNC_TAGS.TAG_UPLOAD]
  },
  {
    fileValue: 'flame.mode.horizontal.syncfl.top_aggregator',
    funcTags: [FUNC_TAGS.TAG_DISTRIBUTE, FUNC_TAGS.TAG_AGGREGATE]
  },
  {
    fileValue: 'flame.mode.horizontal.syncfl.trainer',
    funcTags: [FUNC_TAGS.TAG_FETCH, FUNC_TAGS.TAG_UPLOAD]
  },
  {
    fileValue: 'flame.mode.horizontal.coord_syncfl.top_aggregator',
    funcTags: [FUNC_TAGS.TAG_DISTRIBUTE, FUNC_TAGS.TAG_AGGREGATE]
  },
  {
    fileValue: 'flame.mode.horizontal.coord_syncfl.middle_aggregator',
    funcTags: [FUNC_TAGS.TAG_DISTRIBUTE, FUNC_TAGS.TAG_AGGREGATE, FUNC_TAGS.TAG_FETCH, FUNC_TAGS.TAG_UPLOAD]
  },
  {
    fileValue: 'flame.mode.horizontal.coord_syncfl.trainer',
    funcTags: [FUNC_TAGS.TAG_FETCH, FUNC_TAGS.TAG_UPLOAD]
  },
  {
    fileValue: 'flame.mode.horizontal.coord_syncfl.coordinator',
    funcTags: [FUNC_TAGS.TAG_COORDINATE_WITH_TOP_AGG, FUNC_TAGS.TAG_COORDINATE_WITH_MID_AGG, FUNC_TAGS.TAG_COORDINATE_WITH_TRAINER]
  },
];

