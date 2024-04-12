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

import { useEffect, useState } from 'react'
import ReactFlow, { Background, Edge, Node } from 'reactflow'
import CustomConnectionLine from '../../../../components/custom-connection-line/CustomConnectionLine'
import { GetRunsPayload, NodeMenuItem, RunViewType } from '../../../../entities/JobDetails'
import { edgeTypes, nodeTypes, connectionLineStyle } from '../../../design-details/constants'
import { fitViewOptions } from '../../JobDetailsPage'
import { getEdges, getNodes, getTasksWithLevelsAndCounts } from '../../utils';
import '../../../../components/custom-node-no-interaction/customNodeNoInteraction.css';
import { getGraphLayoutedElements } from '../../../utils'
import { Task } from '../../../../entities/Task'
import { useDisclosure } from '@chakra-ui/react'
import TaskLogs from '../task-logs/TaskLogs'
import { MENU_OPTIONS_PROPERTIES } from '../../constants'

const initialSearchCriteria: Partial<GetRunsPayload> = {
  max_results: 100,
  order_by: ['attributes.start_time DESC'],
  run_view_type: RunViewType.activeOnly,
}

interface Props {
  tasks: Task[] | undefined;
  experiment: any;
  runs: any | undefined;
  mutate: (data: Partial<GetRunsPayload>) => void;
}

const JobTopology = ({ tasks, experiment, runs, mutate }: Props) => {
  const [nodes, setNodes] = useState<Node<any, string | undefined>[]>([]);
  const [edges, setEdges] = useState<Edge<any>[] | undefined>([]);
  const [taskName, setTaskName] = useState<string>('');
  const [taskLog, setTaskLog] = useState<string>('');
  const { isOpen, onOpen, onClose } = useDisclosure();
  const [nodeMenuItems, setNodeMenuItems] = useState<NodeMenuItem[]>();

  useEffect(() => {
    setNodeMenuItems([]);
  }, [tasks])


  const onLogsDisplay = (data: { taskName: string, tasks: Task[] }) => {
    const taskLog = data.tasks?.find(task => task.role === data.taskName)?.log || '';
    setTaskName(data.taskName);
    setTaskLog(taskLog);
    onOpen();
  }

  useEffect(() => {
    if (tasks?.length) {
      const edges = getEdges(tasks);
      const nodes = getNodes(
        getTasksWithLevelsAndCounts(tasks).map(task => ({
          ...task,
          menuItems: getNodeMenuOptions(task)
        }))
      );
      const layouted = getGraphLayoutedElements(nodes, edges, 'TB', 200, 200);

      setEdges([...layouted.edges]);
      setNodes([...layouted.nodes]);
    }
  }, [tasks, runs, nodeMenuItems]);

  useEffect(() => {
    if (!experiment) { return; }
    mutate({
      ...initialSearchCriteria,
      experiment_ids: [experiment?.experiment?.experiment_id || ''],
    })
  }, [experiment]);

  const getNodeMenuOptions = (task: Task) => {
    const menuOptions: NodeMenuItem[] = [];
    if (task[MENU_OPTIONS_PROPERTIES.log as keyof Task]) {
      menuOptions.push({ label: 'Display logs', tasks, callback: onLogsDisplay, propertyName: 'log' })
    }

    return menuOptions;
  }

  return (
    <>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        edgeTypes={edgeTypes}
        nodeTypes={nodeTypes}
        connectionLineComponent={CustomConnectionLine as unknown as any}
        connectionLineStyle={connectionLineStyle}
        fitViewOptions={fitViewOptions}
        fitView
      >
        <Background />
      </ReactFlow>

      <TaskLogs isOpen={isOpen} onClose={onClose} taskName={taskName} log={taskLog} />
    </>
  )
}

export default JobTopology