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

import { useCallback } from 'react';
import { useStore, getStraightPath, EdgeLabelRenderer } from 'reactflow';
import { getEdgeParams } from '../../features/design-details/utils';
import '../../features/design-details/animations.css';

interface Props {
    id: string,
    source: string,
    target: string,
    style: { strokeWidth: number, stroke: string },
    label: string,
}

const FloatingEdge = ({ id, source, target, style, label }: Props) => {
  const sourceNode = useStore(useCallback((store) => store.nodeInternals.get(source), [source]));
  const targetNode = useStore(useCallback((store) => store.nodeInternals.get(target), [target]));

  if (!sourceNode || !targetNode) {
    return null;
  }

  const { sx, sy, tx, ty } = getEdgeParams(sourceNode, targetNode);

  const [edgePath, labelX, labelY] = getStraightPath({
    sourceX: sx,
    sourceY: sy,
    targetX: tx,
    targetY: ty,
  });

  return (
    <>
      <EdgeLabelRenderer>
      <div
          style={{
            position: 'absolute',
            transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
            background: 'white',
            padding: 5,
            borderRadius: 5,
            fontSize: 12,
            fontWeight: 700,
          }}
          className="nodrag nopan" //error-pulse
        >
          {label}
        </div>
      </EdgeLabelRenderer>
      <path
        id={id}
        className="react-flow__edge-path"
        d={edgePath}
        style={style}
      />
    </>
  );
}

export default FloatingEdge;
