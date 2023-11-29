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

import { BaseEdge, BezierEdge, EdgeLabelRenderer, EdgeProps } from 'reactflow';

const SelfConnecting = (props: EdgeProps) => {
  if (props.source !== props.target) {
    return <BezierEdge {...props} />;
  }

  const { sourceX, sourceY, targetX, targetY, markerEnd } = props;
  const radiusX = (sourceX - targetX) * 0.6;
  const radiusY = 50;
  const edgePath = `M ${sourceX - 5} ${sourceY} A ${radiusX} ${radiusY} 0 1 0 ${
    targetX + 2
  } ${targetY}`;

  return <>
    <BaseEdge path={edgePath} markerEnd={markerEnd} style={props.style} />;

    <EdgeLabelRenderer>
      <div
          style={{
            position: 'absolute',
            transform: `translate(-50%, -50%) translate(${sourceX}px,${sourceY / 2}px)`,
            background: 'white',
            padding: 5,
            borderRadius: 5,
            fontSize: 12,
            fontWeight: 700,
          }}
          className="nodrag nopan" //error-pulse
        >
          {props.label}
        </div>
      </EdgeLabelRenderer>
  </>
}

export default SelfConnecting;
