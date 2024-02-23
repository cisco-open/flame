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

import { Tag } from "@chakra-ui/react";
import { stratify, tree } from "d3-hierarchy";
import { Edge, Node, Position } from "reactflow";
import { Channel, GroupBy, MappedFuncTag, Role, Schema } from "../../entities/DesignDetails";
import { SchemaValidity } from "./DesignDetailsPage";

export const getChannel = (schema: Schema | undefined, edge: Edge): Channel | undefined =>
  schema?.channels?.find(channel => `${channel.pair[0]}-${channel.pair[1]}` === edge.id
  )

export const getRole = (schema: Schema | undefined, node: Node): Role | undefined =>
  schema?.roles?.find(role => role.name === node.id);

export const mapNodes = (schema: Schema, externalDesignId?: string) => schema?.roles?.map((role, index) => ({
  id: role.name,
  data: { label: role.name },
  position: { x: role.name === 'coordinator' ? 500 : 100, y: role.name === 'coordinator' ? 400 : index * 200 },
  type: externalDesignId ? 'customNodeNoInteraction' : 'custom',
  groupAssociation: role.groupAssociation,
  dragHandle: '.custom-drag-handle',
  role,
}));

export const mapEdges = (schema: Schema, externalDesignId?: string) => schema?.channels?.map(channel => ({
  id: `${channel.pair[0]}-${channel.pair[1]}`,
  label: channel.name,
  source: channel.pair[0],
  target: channel.pair[1],
  type: getEdgeType(channel, externalDesignId),
  channel: { ...channel },
}));

const getEdgeType = (channel: Channel, externalDesignId: string | undefined) => {
  if (externalDesignId) {
    return channel.pair[1] === channel.pair[0] ? 'selfConnectingNoInteraction' : 'noInteraction';
  }

  return channel.pair[1] === channel.pair[0] ? 'selfConnecting' : 'floating';
}

export const getEdgeParams = (source: any, target: any) => {
  const sourceIntersectionPoint = getNodeIntersection(source, target);
  const targetIntersectionPoint = getNodeIntersection(target, source);

  const sourcePos = getEdgePosition(source, sourceIntersectionPoint);
  const targetPos = getEdgePosition(target, targetIntersectionPoint);

  return {
    sx: sourceIntersectionPoint.x,
    sy: sourceIntersectionPoint.y,
    tx: targetIntersectionPoint.x,
    ty: targetIntersectionPoint.y,
    sourcePos,
    targetPos,
  };
}

const getNodeIntersection = (intersectionNode: any, targetNode: any) => {
  const {
    width: intersectionNodeWidth,
    height: intersectionNodeHeight,
    positionAbsolute: intersectionNodePosition,
  } = intersectionNode;
  const targetPosition = targetNode.positionAbsolute;

  const w = intersectionNodeWidth / 2;
  const h = intersectionNodeHeight / 2;

  const x2 = intersectionNodePosition.x + w;
  const y2 = intersectionNodePosition.y + h;
  const x1 = targetPosition.x + w;
  const y1 = targetPosition.y + h;

  const xx1 = (x1 - x2) / (2 * w) - (y1 - y2) / (2 * h);
  const yy1 = (x1 - x2) / (2 * w) + (y1 - y2) / (2 * h);
  const a = 1 / (Math.abs(xx1) + Math.abs(yy1));
  const xx3 = a * xx1;
  const yy3 = a * yy1;
  const x = w * (xx3 + yy3) + x2;
  const y = h * (-xx3 + yy3) + y2;

  return { x, y };
}

const getEdgePosition = (node: any, intersectionPoint: any) => {
  const n = { ...node.positionAbsolute, ...node };
  const nx = Math.round(n.x);
  const ny = Math.round(n.y);
  const px = Math.round(intersectionPoint.x);
  const py = Math.round(intersectionPoint.y);

  if (px <= nx + 1) {
    return Position.Left;
  }
  if (px >= nx + n.width - 1) {
    return Position.Right;
  }
  if (py <= ny + 1) {
    return Position.Top;
  }
  if (py >= n.y + n.height - 1) {
    return Position.Bottom;
  }

  return Position.Top;
}

const getDuplicateNames = (nodes: any, edges: any) => {
  const roleNames = nodes?.map((node: any) => node.role).map((role: any) => role.name.toLowerCase()) || [];
  const channelNames = edges?.map((edge: any) => edge.channel).map((channel: any) => channel.name.toLowerCase()) || [];

  const namesDictionary = [...roleNames, ...channelNames].reduce((acc: any, curr: any) => {
    return acc[curr] ? ++acc[curr] : acc[curr] = 1, acc
  }, {});
  return Object.keys(namesDictionary).filter((key: string) => namesDictionary[key] > 1);
}

const ERROR_MESSAGE_MAPPING: { [key: string]: string } = {
  multipleDataConsumerRoles: 'You can only have one data consumer role',
  noGroupOnEachChannel: 'You need to add at least one group per channel',
  codeFileMissing: 'You need to add at least one code file to each role',
  noDataConsumerRoles: 'You need to have at least one data consumer role',
  allRolesNotConnected: 'You need to connect all roles'
}

export const getErrorMessages = (schemaValidity: SchemaValidity) => {
  const errorMessages = Object.keys(schemaValidity)
    .filter((item: any) => !!(schemaValidity as unknown as any)[item])
    .map(item => ERROR_MESSAGE_MAPPING[item]);

  if (schemaValidity.duplicateNames?.length) {
    const isPlural = schemaValidity.duplicateNames.length > 1;
    errorMessages.push(`${isPlural ? 'Names' : 'Name'} "${schemaValidity.duplicateNames.join(', ')}" ${isPlural ? 'are' : 'is'} duplicate`)
  }

  return errorMessages.filter(message => !!message);
}

export const getSchemaValidity = (nodes: any, edges: any, fileNames: any) => {
  const roles = nodes?.map((node: any) => node.role);
  const dataConsumerRoles = roles?.filter((role: Role) => role.isDataConsumer);
  const channels = edges?.map((edge: any) => edge.channel);
  const channelGroups = channels?.map((channel: Channel) => channel.groupBy.value?.filter(value => !!value?.length)).filter((value: string[] | undefined) => !!(value as unknown as any)?.length);
  const duplicateNames = getDuplicateNames(nodes, edges);
  return {
    multipleDataConsumerRoles: dataConsumerRoles?.length > 1,
    noDataConsumerRoles: !dataConsumerRoles?.length,
    noGroupOnEachChannel: channelGroups?.length !== channels?.length,
    codeFileMissing: !fileNames || fileNames?.length < roles?.length,
    allRolesNotConnected: roles?.length === 1 ? !channels?.length : channels?.length < roles?.length - 1,
    duplicateNames
  };
}

export const getUpdatedNodes = (role: any, nodes: any) => {
  if (!nodes.length) { return; }

  const index = role?.index || 0;
  const targetNode = nodes[index];
  const newNode = {
    ...targetNode,
    data: { label: role.name },
    isDataConsumer: role.isDataConsumer,
    role: { ...role },
  }
  const updatedNodes = [...nodes];
  updatedNodes[index] = newNode;
  return updatedNodes;
}

export const getUpdatedEdges = (channel: Channel, edges: Edge[]) => {
  if (!edges?.length) { return; }
  const index = channel?.index || 0;
  const targetEdge = edges[index];
  const newEdge = {
    ...targetEdge,
    label: channel.name,
    channel: {
      ...channel
    }
  }
  const updatedEdges = [...edges];
  updatedEdges[index] = newEdge;
  return updatedEdges;
}

export const getChannelsFromEdges = (channels: Channel[] | undefined, edges: Edge[]): Channel[] => {
  return edges.map(edge => {
    const channel = channels?.find(channel => JSON.stringify(channel.pair.sort()) === JSON.stringify([edge.source, edge.target].sort()));
    return {
      description: channel ? channel.description : '',
      funcTags: channel ? channel.funcTags : { aggregator: [], trainer: [] },
      groupBy: channel ? channel.groupBy : { type: '', value: [] },
      name: channel ? channel.name : '',
      pair: channel ? channel.pair : [edge.source, edge.target],
    }
  });
};

export const getDefaultChannel = (source: string, target: string, numberOfEdges: number) => {
  return {
    description: '',
    funcTags: {},
    groupBy: { type: 'tag', value: ['default']},
    name: `channel-${numberOfEdges}`,
    pair: [source, target],
  }
}

export const getDefaultNode = (length: number) => {
  const name = `Role-${(length) + 1}`;
  return {
    id: name,
    name,
    description: '',
    isDataConsumer: false,
    dragHandle: '.custom-drag-handle',
    position: { x: 100, y: (length + 1) * 100 },
    data: {
      id: name,
      label: name,
    },
    type: 'custom',
    role: {
      name,
      description: '',
      groupAssociation: [],
      isDataConsumer: false,
    }
  }
}

export const getGroupBy = (groupBy: string): GroupBy => {
  return {
    type: 'tag',
    value: groupBy.split(',').map(value => value.trim()),
  }
}

export const updateRolesByEdges = (edges: any, nodes: any) => {
  return nodes.map((node: any) => {
    const includedChannels: any[] = edges
      .filter((edge: any) => edge.channel.pair.includes(node.role.name))
      .map((edge: any) => ({
        groupByValue: (edge.channel.groupBy?.value || []).map((value: string) => ({
          [edge.channel.name]: value,
        })),
        name: edge.channel.name
      }))
      .filter((edge: any) => edge?.groupByValue?.length)
      .sort((a: any, b: any) => b.groupByValue.length - a.groupByValue.length);

    const referenceChannel = includedChannels[0];
    let result = referenceChannel?.groupByValue;
    for (let i = 1; i < includedChannels.length; i++) {
      const current = includedChannels[i];
      const combined = [];

      for (const item1 of result) {
        for (const item2 of current.groupByValue) {
          combined.push({ ...item1, ...item2 });
        }
      }

      result = combined;
    }

    return {
      ...node,
      role: {
        ...node.role,
        groupAssociation: result,
      }
    };
  })
}

export const getUpdatedEdgesByRole = (role: any, edges: any[]) => {
  return edges?.map((edge: any) => {
    const index = edge.channel.pair.findIndex((value: string) => value === role.previousName);
    if (index !== -1) {
      edge.channel.pair.splice(index, 1, role.name);
    }
    return {
      ...edge
    }
  });
}

export const getMappedFuncTags = (channel: Channel, funcTags: MappedFuncTag[], channels: Channel[]): any[] => {
  return funcTags.filter(tag => channel.pair.includes(tag.roleName)).map(tag => {
    const targetChannelFuncTag = channel?.funcTags?.[tag.roleName];

    if (!targetChannelFuncTag) {
      return;
    }

    const otherChannelsFuncTags = channels?.filter(ch =>
      ch?.funcTags &&
      ch.name !== channel.name &&
      Object.keys(ch.funcTags).includes(tag.roleName) &&
      ch.pair.includes(tag.roleName)
    ).map(ch => ch.funcTags[tag.roleName]).reduce((acc: string[], current) => [...acc, ...current], []);

    return {
      ...tag,
      funcTags: tag.funcTags.map(tag => ({
        value: tag.value,
        selected: targetChannelFuncTag.includes(tag.value) && !otherChannelsFuncTags.includes(tag.value),
        disabled: otherChannelsFuncTags.includes(tag.value),
      }))
    }
  }).filter(tag => !!tag);
}

export const getFuncTagsFromFile = (funcTags: MappedFuncTag[], channel: any) => {
  if (!funcTags?.length) { return; }
  return funcTags?.filter(tag => channel.pair.includes(tag.roleName));
}

export const setFuncTagsToChannels = (edges: any[], funcTags: any) => {
  if (!funcTags?.length) { return; }
  return edges.map(edge => {
    return {
      ...edge,
      channel: {
        ...edge.channel,
        funcTags: getFuncTagsFromFile(funcTags, edge.channel)?.reduce((acc, item) => ({
          ...acc,
          [item.roleName]: [...item.funcTags],
        }), {})
      }
    }
  });
}

export const addEdgeWithDefaultChannel = (params: any, eds: any) => {
  let type = 'floating';

  if (params.source === params.target) {
    type = 'selfConnecting';
  }
  return {
    ...params,
    type,
    label: `channel-${(eds?.length || 0) + 1}`,
    channel: getDefaultChannel(params.source, params.target, (eds?.length || 0) + 1)
  };
}

export const createCodeFileData = (fileData: any, data: any) => {
  let newFiles: any[];
  const index = fileData.findIndex((file: any) => file.roleName === data.role.name);
  if (index === -1) {
    newFiles = [...fileData, { file: data.file, roleName: data.role.name }];
  } else {
    fileData.splice(index, 1);
    newFiles = [...fileData, { file: data.file, roleName: data.role.name }];
  }

  return newFiles;
}

export const getChannelPayload = (formValue: any, channel: any, funcTags: MappedFuncTag[] | undefined) => {
  const mappedFuncTags = funcTags?.reduce((acc, item) => ({
    ...acc,
    [item.roleName]: [...item.funcTags.filter(tag => tag.selected).map(tag => tag.value)],
  }), {});

  return {
    ...channel,
    ...formValue,
    groupBy: getGroupBy(formValue.groupBy),
    funcTags: mappedFuncTags,
  }
}

export const getSchemaPayload = (designSchema: any, nodes: any, edges: any, id: string | undefined) => ({
  ...designSchema,
  roles: [...nodes.map((node: any) => {
    const { name, groupAssociation, isDataConsumer, replica } = node.role;
    return {
      name,
      groupAssociation,
      isDataConsumer,
      replica
    }
  })],
  channels: [...(edges as unknown as any[]).map(edge => {
    const { description, funcTags, groupBy, name, pair } = edge.channel;
    return {
      description,
      funcTags,
      groupBy,
      name,
      pair,
    }
  })],
  name: id
});

export const createDesignCodeZip = (fileData: any, id: string | undefined) => {
  const jsZip = require("jszip");
  const zip = new jsZip();
  for (let i = 0; i < fileData.length; i++) {
    const mainFolder = zip.folder(`${fileData[i].roleName}`, { binary: true });
    mainFolder.file(`${fileData[i].file.name}`, fileData[i].file, { binary: true });
  }

  return zip;
}

export const getEdgesForExpanded = (nodes: any) => {
  const pairs: any[] = [];
  const mappedNodes = nodes.map((node: any) => {
    return {
      ...node,
      count: nodes.filter((n: any) => n.role.name === node.role.name).length
    }
  }).sort(((a: any, b: any) => a.count - b.count));;
  for (let i = 1; i < mappedNodes.length; i++) {
    const currentNode = mappedNodes[i];
    // find the pair of the current node
    let pair: any;
    if (currentNode.replicaId) { // current node is a replica and it has to connect to the same trainer 
      pair = mappedNodes.filter((n: any) => {
        return n.parentId === currentNode.replicaId ||
        (
          !n.role.isDataConsumer && n.role.name !== currentNode.role.name &&
          n.id !== currentNode.id &&
          hasSameGroup(n, currentNode)
        )
      })
    } else { // other nodes that are not replicas
      pair = mappedNodes.find((n: any) => {
        if (currentNode.parentId) {
          return n.id === currentNode.parentId;
        }
        return !n.role.isDataConsumer && n.role.name !== currentNode.role.name &&
          n.id !== currentNode.id &&
          hasSameGroup(n, currentNode)
      });
    }



    if (Array.isArray(pair)) { // create pairs for replica
      for (const p of pair) {
        pairs.push([
          p,
          currentNode,
        ]);
      }
    } else { // create pairs for every other node
      if (pair) {
        pairs.push([
          pair,
          currentNode,
        ]);
      }
    }
  }
  // map pairs to source - target edges
  return pairs.map((pair: any[]) => {
    const isPlaceholderNode = pair.find((p: any) => p.placeholderNode);
    return {
      id: `${pair[0].id}-${pair[1].id}`,
      source: pair[0].id,
      target: pair[1].id,
      type: isPlaceholderNode ? 'invisible' : 'floating',
      label: isPlaceholderNode ? '' : `${getEdgeLabel(pair[0], pair[1])}`,
    }
  })
}

const getEdgeLabel = (first: any, second: any) => {
  return first.group?.find((group: any) => second.group?.includes(group));
}

const hasSameGroup = (node: any, secondNode: any) => {
  return !!node.group?.find((group: string) => secondNode.group?.includes(group));
}

export const getNodesForExpandedTopology = (nodes: any) => {
  const newNodes = [];
  const dataNodes: any[] = [];
  const nonDataConsumerNodes = nodes.filter((node: any) => !node.role.isDataConsumer).map((node: any) => ({
    ...node,
    group: Object.values(node.role.groupAssociation)
  }));
  const dataConsumerNodes = nodes.filter((node: any) => node.role.isDataConsumer).map((node: any) => ({
    ...node,
    group: node.role.groupAssociation.map((group: any) => Object.values(group).join(''))
  }));

  // create nodes for non data consumer roles
  for (const node of nonDataConsumerNodes) {
    for (let i = 0; i < node.role.groupAssociation.length; i++) {
      newNodes.push({
        ...node,
        id: `${node.id}${i}`,
        group: Object.values(node.role.groupAssociation[i]),
        type: 'customNodeNoInteraction',
      });
    }
  }

  // create nodes for replica
  for (let node of [...newNodes]) {
    if (node.role.replica > 1) {
      for (let r = 0; r < node.role.replica - 1; r++) {
        newNodes.push({
          ...node,
          id: `${node.id}${r}`,
          replicaId: `${node.id}`,
        });
      }
    }
  }

  // create 3 nodes for data consumer roles for each parent node -> node ... node
  for (let node of dataConsumerNodes) {
    const parentNodes = newNodes.filter((parentNode: any) => hasSameGroup(parentNode, node) && !parentNode.replicaId);
    parentNodes.forEach((parentNode: any, index: number) => {
      dataNodes.push(...[
        {
          ...node,
          id: `data-consumer-${parentNode.id}${Math.random()}`,
          parentId: parentNode.id,
        },
        {
          ...node,
          id: `data-consumer-${parentNode.id}${Math.random()}`,
          data: { label: '...'},
          type: 'placeholder',
          placeholderNode: true,
          parentId: parentNode.id,
        },
        {
          ...node,
          id: `data-consumer-${parentNode.id}${Math.random()}`,
          parentId: parentNode.id,
        }
      ])
    })
  }
  return [...newNodes, ...dataNodes];
}

export const getSortedNodes = (nodes: any) => {
  return nodes.sort((a: any, b: any) => {
    if (a.parentId || b.parentId) {
      return 0;
    }
    if ( a.id < b.id ){
      return -1;
    }
    if ( a.id > b.id ){
      return 1;
    }
    return 0;
  })
}

export const getTreeLayoutedElements = (nodes: any, edges: any) => {
  if (nodes.length === 0) return { nodes, edges };
  const g = tree();

  const sortedNodes = getSortedNodes(nodes);
  const hierarchy = stratify()
    .id((node: any) => node.id)
    .parentId((node: any) => edges.find((edge: any) => edge.target === node.id)?.source);
  const root = hierarchy(sortedNodes);
  const layout = g.nodeSize([100 * 2, 100 * 2]).separation((a, b) => 1)(root);

  return {
    nodes: layout
      .descendants()
      .map((node) => ({ ...(node as unknown as any).data, position: { x: node.x, y: node.y } })),
    edges,
  };
};