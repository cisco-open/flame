import { Handle, Position, ReactFlowState, useStore } from 'reactflow';
import { useEffect, useState } from 'react';
import { Tooltip } from '@chakra-ui/react';

const colors: any = {
  failed: 'red',
  completed: 'gray',
  ready: 'green',
  terminated: 'red'
}

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
    setStatusColor(colors[status])
    setTooltip(`${status?.[0].toUpperCase()}${status?.substring(1)}`)
  }, [status])

  return (
    <div className={`customNode ${isInteractive ? 'interactive' : 'no-interaction'}`}>
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
