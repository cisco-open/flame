import { getStraightPath } from 'reactflow';

interface Props {
    fromX: number,
    fromY: number,
    toX: number,
    toY: number,
    connectionLineStyle?: { strokeWidth: number, stroke: string };
}

const CustomConnectionLine = ({ fromX, fromY, toX, toY, connectionLineStyle }: Props) => {
  const [edgePath] = getStraightPath({
    sourceX: fromX,
    sourceY: fromY,
    targetX: toX,
    targetY: toY,
  });

  return (
    <g>
      <path style={connectionLineStyle} fill="none" d={edgePath} />
      <circle cx={toX} cy={toY} fill="black" r={3} stroke="black" strokeWidth={2} />
    </g>
  );
}

export default CustomConnectionLine;
