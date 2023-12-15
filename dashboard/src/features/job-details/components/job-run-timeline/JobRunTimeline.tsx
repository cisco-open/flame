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

import { Modal, ModalOverlay, ModalContent, ModalCloseButton, ModalBody, Box, ModalHeader} from '@chakra-ui/react'
import { useEffect, useState } from 'react'
import { Run, RunResponse } from '../../../../entities/JobDetails'
import { Task } from '../../../../entities/Task'
import Loading from '../../../../layout/loading/Loading'
import ApiClient from '../../../../services/api-client'
import { getRuntimeMetrics } from '../../utils'
import MetricsTimeline from '../metrics-timeline/MetricsTimeline'

interface Props {
  isOpen: boolean,
  runsResponse: RunResponse | undefined,
  runs: any;
  tasks: Task[] | undefined,
  jobName: string,
  onClose: () => void;
}

interface UiMetric {
  name: string,
  from: number,
  to: number,
  time: number,
  category: string,
  stackId: string,
  runId: string,
  id: string,
  fill: string,
}

const JobRunTimeline = ({ isOpen, runsResponse, runs, jobName, tasks, onClose }: Props) => {
  const [runtimes, setRuntimes] = useState<any[]>([]);
  const [loading, setIsLoading] = useState<boolean>(true);
  const [names, setNames] = useState<any[]>([]);
  const [responses, setResponses] = useState<any[]>([]);
  const [transformedData, setTransformedData] = useState<any>();

  useEffect(() => {
    if (!runtimes.length) { return; }

    fetchMetrics(runtimes, names);
  }, [runtimes, names]);

  useEffect(() => {
    if (!Object.keys(responses)?.length) { return; }
    const data: any = {}

    Object.keys(responses).map((key: any) => (
      data[key] = responses[key]
        .reduce((acc: any[], curr: any) => [...acc, ...curr], [])
        .map((item: any, index: number, list: any[]) => {
          if (!item.key.includes('runtime')) { return item; }
          const startTimeCounterpart = list.find(i => i.key.includes(item.key.split('.')[0]) && i.key.includes('starttime') && item.step === i.step);
          const startTimeTimestamp = Math.round(startTimeCounterpart.value * 1000);
          return {
            ...item,
            start: startTimeTimestamp,
            category: key
          }
        })
        .filter((item: any) => item.key.includes('runtime'))
    ));

    setIsLoading(false);
    setTransformedData(data);
  }, [responses])

  useEffect(() => {
    if (!tasks && !runsResponse) { return; }
    const taskIds = tasks?.map(task => task.taskId.substring(0, 8));

    const runs = runsResponse?.runs.filter((run, index, runs) =>
      run.info.start_time &&
      run.info.end_time &&
      taskIds?.includes(run.info.run_name.substring(run.info.run_name.length - 8)) &&
      hasTheGreatestTimestamp(run, runs)
    );
    const mappedData = getRuntimeMetrics(runs);
    setRuntimes(mappedData.runtimes);
    setNames(mappedData.names);
    setIsLoading(true);
  }, [tasks, runsResponse]);

  const fetchMetrics = async (runtimes: UiMetric[], names: string[]) => {
    const fetchedResponses: any = names.reduce((acc, name) => ({ ...acc, [name]: []}), {});
    const apiClient = new ApiClient('mlflow/metrics/get-history', true);

    for (const element of runtimes) {
      const { name, runId, category } = element;


      try {
        const response = await apiClient.getAll({ params: { metric_key: name, run_uuid: runId, }});
        fetchedResponses[category] = [...fetchedResponses[category], (response as unknown as any).metrics];
      } catch (error) {
        console.error('Error fetching metrics:', error);
      }
    }

    const newData: any = {}
    Object.keys(fetchedResponses).map((name: string) => {
      const transformedData: any = {};
      fetchedResponses[name] = fetchedResponses[name].map((stepArray: any) => {
        stepArray.forEach((item: any) => {
          const { step } = item;
          if (!transformedData[step]) {
            transformedData[step] = [];
          }
          transformedData[step].push(item);
        });
        newData[name] = {...newData[name], ...transformedData };
        return transformedData;
      });
    });

    Object.keys(newData).map((key: string) => {
      newData[key] = Object.keys(newData[key]).map(k => newData[key][k])
    });
    setResponses(newData);
  };

  const hasTheGreatestTimestamp = (run: Run, runs: Run[]) => {
    const targetRuns = runs.filter(r => r.info.run_name === run.info.run_name && run.info.start_time && run.info.end_time);
    let targetRun = targetRuns[0];
    for (let run of targetRuns) {
      if (targetRun.info.start_time < run.info.start_time) {
        targetRun = run;
      }
    }

    return targetRun.info.run_id === run.info.run_id;
  }

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      size="full"
    >
      <ModalOverlay />

      <ModalContent className="dataset-form" height="100%">
        <ModalHeader textAlign="center">{jobName} runtime metrics</ModalHeader>

        <ModalCloseButton />

        <ModalBody pb={6} display="flex" height="100%" overflow="hidden" overflowY="auto" flexDirection="column" gap="10px">
          {
            loading ?
            <Box height="100%" display="flex" justifyContent="center" alignItems="center">
              <Loading message={'Fetching data...'} />
            </Box> :
            <MetricsTimeline data={transformedData} runs={runs}/>
          }
        </ModalBody>
      </ModalContent>
    </Modal>
  )
}

export default JobRunTimeline