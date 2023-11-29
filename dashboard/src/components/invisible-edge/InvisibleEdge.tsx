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
import { useStore } from 'reactflow';
import { getEdgeParams } from '../../features/design-details/utils';
import '../../features/design-details/animations.css';

interface Props {
    id: string,
    source: string,
    target: string,
    style: { strokeWidth: number, stroke: string },
    label: string,
}

const InvisibleEdge = ({ id, source, target, style }: Props) => {
  const sourceNode = useStore(useCallback((store) => store.nodeInternals.get(source), [source]));
  const targetNode = useStore(useCallback((store) => store.nodeInternals.get(target), [target]));

  if (!sourceNode || !targetNode) {
    return null;
  }

  const { sx, sy, tx, ty } = getEdgeParams(sourceNode, targetNode);

  return (
    <>
      <path
        id={id}
        style={style}
      />
    </>
  );
}

export default InvisibleEdge;
