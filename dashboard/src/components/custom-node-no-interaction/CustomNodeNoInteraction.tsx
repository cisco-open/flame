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

import { Handle, Position, ReactFlowState, useStore } from 'reactflow';
import { useEffect, useState } from 'react';
import { Tooltip } from '@chakra-ui/react';
import { NODE_COLORS } from '../../constants';

const connectionNodeIdSelector = (state: ReactFlowState) => state.connectionNodeId;

interface Props {
    data: { id: string, status: string, label: string, isInteractive: boolean };
}

const CustomNodeNoInteraction = ({ data: { id, status, label, isInteractive } }: Props) => {
  const connectionNodeId = useStore(connectionNodeIdSelector);
  const [ statusColor, setStatusColor ] = useState('green');
  const [ tooltip, setTooltip ] = useState('green');
  const isConnecting = !!connectionNodeId;

  useEffect(() => {
    setStatusColor(NODE_COLORS[status])
    setTooltip(`${status?.[0].toUpperCase()}${status?.substring(1)}`)
  }, [status])

  return (
    <div className={`customNode ${isInteractive ? 'interactive interactive-pulse' : 'no-interaction'}`}>
      <div className="customNodeBody">
        {!isConnecting && (
          <Handle
            className="custom-node-no-interaction-handle"
            position={Position.Right}
            type="source"
          />
        )}

        {
          status &&
          <Tooltip label={tooltip} fontSize="inherit">
            <span
              className="status-indicator"
              style={{
                backgroundColor: statusColor
              }}
            ></span>
          </Tooltip>
        }

        <Handle isConnectable={false} className="customHandle" position={Position.Left} type="target"></Handle>
        {label}
      </div>
    </div>
  );
}

export default CustomNodeNoInteraction;
