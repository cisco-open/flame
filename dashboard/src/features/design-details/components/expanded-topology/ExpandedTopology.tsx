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

import { useEffect, useState } from 'react';
import ReactFlow, { Background, useReactFlow } from 'reactflow';
import CustomConnectionLine from '../../../../components/custom-connection-line/CustomConnectionLine';
import { edgeTypes, nodeTypes, defaultEdgeOptions, connectionLineStyle } from '../../constants';
import { getEdgesForExpanded, getTreeLayoutedElements, getNodesForExpandedTopology } from '../../utils';


interface Props {
  nodes: any;
}

const ExpandedTopology = ({ nodes }: Props) => {
  const { fitView } = useReactFlow();
  const [ newNodes, setNewNodes ] = useState<any[]>([]);
  const [ edges, setEdges ] = useState<any[]>([]);

  useEffect(() => {
    const newNodes = getNodesForExpandedTopology(nodes);
    const edges = getEdgesForExpanded(newNodes);

    const { nodes: layoutedNodes, edges: layoutedEdges } = getTreeLayoutedElements(newNodes, edges);

    setNewNodes([...layoutedNodes]);
    setEdges([...layoutedEdges]);

    window.requestAnimationFrame(() => {
      fitView();
    });
  }, [nodes])

  return (
    <ReactFlow
      nodes={newNodes}
      edges={edges}
      edgeTypes={edgeTypes}
      nodeTypes={nodeTypes}
      defaultEdgeOptions={defaultEdgeOptions}
      connectionLineComponent={CustomConnectionLine as unknown as any}
      connectionLineStyle={connectionLineStyle}
      fitView
    >
      <Background />
    </ReactFlow>
  )
}

export default ExpandedTopology;