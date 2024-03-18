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
import { Box, Button, Popover, PopoverBody, PopoverContent, PopoverTrigger, Tooltip } from '@chakra-ui/react';
import { NODE_COLORS } from '../../constants';
import { NodeMenuItem } from '../../entities/JobDetails';
import { FaEllipsisVertical } from 'react-icons/fa6';

const connectionNodeIdSelector = (state: ReactFlowState) => state.connectionNodeId;

interface Props {
    data: {
      id: string,
      status: string,
      label: string,
      log: string,
      isInteractive: boolean,
      menuItems?: NodeMenuItem[],
    };
}

const CustomNodeNoInteraction = ({ data: { status, label, isInteractive, menuItems } }: Props) => {
  const connectionNodeId = useStore(connectionNodeIdSelector);
  const [ statusColor, setStatusColor ] = useState('green');
  const [ tooltip, setTooltip ] = useState('green');
  const isConnecting = !!connectionNodeId;

  useEffect(() => {
    setStatusColor(NODE_COLORS[status])
    setTooltip(`${status?.[0].toUpperCase()}${status?.substring(1)}`)
  }, [status])

  return (
    <div onWheelCapture={(event) => { event.preventDefault(); event?.stopPropagation(); }} onWheel={(event) => { event.preventDefault(); event?.stopPropagation(); }} className={`customNode ${isInteractive ? 'interactive interactive-pulse' : 'no-interaction'}`}>
      <div className="customNodeBody">
        {!isConnecting && (
          <Handle
            className="custom-node-no-interaction-handle"
            position={Position.Right}
            type="source"
          />
        )}

        {
          !!menuItems?.length &&
          <Popover>
            <PopoverTrigger>
              <Button className='menu-button' leftIcon={<FaEllipsisVertical />} />
            </PopoverTrigger>

            <PopoverContent>
              <PopoverBody>
                {
                  menuItems.map(item =>
                    <Box className="custom-node-no-interaction-menu-item" key={item.label} onClick={() => { item.callback({ taskName: label, tasks: item.tasks }) }}>
                    { item.label }
                    </Box>
                  )
                }
              </PopoverBody>
            </PopoverContent>
          </Popover>
        }


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

        <Handle isConnectable={false} className="customHandle custom-node-no-interaction-handle__target" position={Position.Left} type="target"></Handle>
        {label}
      </div>
    </div>
  );
}

export default CustomNodeNoInteraction;
