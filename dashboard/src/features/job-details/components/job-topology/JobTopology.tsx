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

import { useDisclosure } from '@chakra-ui/react'
import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import ReactFlow, { Background, Edge, Node } from 'reactflow'
import CustomConnectionLine from '../../../../components/custom-connection-line/CustomConnectionLine'
import { GetRunsPayload, Run, RunViewType } from '../../../../entities/JobDetails'
import { edgeTypes, nodeTypes, connectionLineStyle } from '../../../design-details/constants'
import useTasks from '../../../jobs/hooks/useTasks'
import useExperiment from '../../hooks/useExperiment'
import useRuns from '../../hooks/useRuns'
import { fitViewOptions } from '../../JobDetailsPage'
import { getEdges, getNodes, getTasksWithLevelsAndCounts } from '../../utils';
import RunDetailsModal from '../run-details-modal/RunDetailsModal';
import '../../../../components/custom-node-no-interaction/customNodeNoInteraction.css';
import { getLayoutedElements } from '../../../utils'

const initialSearchCriteria: Partial<GetRunsPayload> = {
  max_results: 100,
  order_by: ['attributes.start_time DESC'],
  run_view_type: RunViewType.activeOnly,
}

const JobTopology = () => {
  const { id } = useParams();
  const [nodes, setNodes] = useState<Node<any, string | undefined>[]>([]);
  const [edges, setEdges] = useState<Edge<any>[] | undefined>([]);
  const [runs, setRuns] = useState<Run[] | undefined>();
  const { data: tasks } = useTasks(id || '');
  const [runDetails, setRunDetails] = useState<Run | undefined>(undefined);
  const { data: runsResponse, mutate } = useRuns();
  const { data: experiment } = useExperiment(id || '');
  const { isOpen, onOpen, onClose } = useDisclosure();

  useEffect(() => {
    if (tasks?.length) {
      const edges = getEdges(tasks);
      const nodes = getNodes(getTasksWithLevelsAndCounts(tasks), runs);
      const layouted = getLayoutedElements(nodes, edges, 'TB', 200, 200);

      setEdges([...layouted.edges]);
      setNodes([...layouted.nodes]);
    }
  }, [tasks, runs]);

  useEffect(() => {
    if (!experiment) { return; }
    mutate({
      ...initialSearchCriteria,
      experiment_ids: [experiment?.experiment?.experiment_id || ''],
    })
  }, [experiment]);

  useEffect(() => {
    const runs = runsResponse?.runs.map(run => {
      const runNameSlices = run.info.run_name.split('-');
      const taskId = runNameSlices[runNameSlices.length - 1];
      return {
        ...run,
        taskId,
      }
    });
    setRuns(runs);
  }, [runsResponse]);

  const onNodeClick = (event: React.MouseEvent, node: Node) => {
    event.stopPropagation();
    const selectedRun = runs?.find(run => node.data.id.includes(run.taskId));
    setRunDetails(selectedRun);

    if (selectedRun) {
      onOpen();
    }
  }

  return (
    <>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        edgeTypes={edgeTypes}
        nodeTypes={nodeTypes}
        onNodeClick={onNodeClick}
        connectionLineComponent={CustomConnectionLine as unknown as any}
        connectionLineStyle={connectionLineStyle}
        fitViewOptions={fitViewOptions}
        fitView
      >
        <Background />
      </ReactFlow>

      <RunDetailsModal isOpen={isOpen} onClose={onClose} runDetails={runDetails}/>
    </>
  )
}

export default JobTopology