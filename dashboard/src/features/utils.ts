import Dagre from '@dagrejs/dagre';

export const getLayoutedElements = (nodes: any[], edges: any[], rankdir: string, ranksep: number, nodesep: number) => {
  const g = new Dagre.graphlib.Graph().setDefaultEdgeLabel(() => ({}));

  g.setGraph({ rankdir, ranksep, nodesep, });

  edges.forEach((edge) => g.setEdge(edge.source, edge.target));
  nodes.forEach((node: any) => g.setNode(node.id, node));

  Dagre.layout(g);

  return {
    nodes: nodes.map((node: any) => {
      const { x, y } = g.node(node.id);

      return { ...node, position: { x, y } };
    }),
    edges,
  };
};
