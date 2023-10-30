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
