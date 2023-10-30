import { useEffect, useState } from 'react';
import ReactFlow, { Background, useReactFlow } from 'reactflow';
import CustomConnectionLine from '../../../../components/custom-connection-line/CustomConnectionLine';
import { edgeTypes, nodeTypes, defaultEdgeOptions, connectionLineStyle } from '../../constants';
import { getEdgesForExpanded, getLayoutedElements, getNodesForExpandedTopology } from '../../utils';


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

    const { nodes: layoutedNodes, edges: layoutedEdges } = getLayoutedElements(newNodes, edges);

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