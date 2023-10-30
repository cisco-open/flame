import { Handle, Position, ReactFlowState, useStore } from 'reactflow';
import { useEffect, useState } from 'react';
import { Tooltip } from '@chakra-ui/react';
import '../../features/design-details/animations.css';

const colors: any = {
  failed: 'red',
  completed: 'gray',
  ready: 'green',
  terminated: 'red'
}

const connectionNodeIdSelector = (state: ReactFlowState) => state.connectionNodeId;

const sourceStyle = { zIndex: 1 };

interface Props {
    data: { id: string, status: string, label: string };
}

const CustomNode = ({ data: { id, status, label } }: Props) => {
  const connectionNodeId = useStore(connectionNodeIdSelector);
  const [ statusColor, setStatusColor ] = useState('green');
  const [ tooltip, setTooltip ] = useState('green');
  const isConnecting = !!connectionNodeId;
  const isTarget = connectionNodeId && connectionNodeId !== id;

  useEffect(() => {
    setStatusColor(colors[status])
    setTooltip(`${status?.[0].toUpperCase()}${status?.substring(1)}`)
  }, [status])

  return (
    <div className="custom-node">
      <div
        className="custom-node-body " //error-pulse
        style={{
          borderStyle: isTarget ? 'dashed' : 'solid',
          backgroundColor: isTarget ? '#ffcce3' : '#ccd9f6',
        }}
      >
        {!isConnecting && (
          <Handle
            className="custom-handle"
            position={Position.Right}
            type="source"
            style={sourceStyle}
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

        <Handle className="custom-handle" position={Position.Left} type="target" />
        <span className="custom-drag-handle">{label}</span>
      </div>
    </div>
  );
}

export default CustomNode;
