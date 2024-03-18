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

import { Modal, ModalOverlay, ModalContent, ModalCloseButton, ModalBody, Box, ModalHeader, Tooltip, Text, Popover, PopoverBody, PopoverContent, PopoverTrigger, UnorderedList} from '@chakra-ui/react'
import { useEffect, useRef, useState } from 'react'
import { RunResponse } from '../../../../entities/JobDetails'
import { Task } from '../../../../entities/Task'
import Loading from '../../../../layout/loading/Loading'
import ApiClient from '../../../../services/api-client'
import { getLatestRuns, getRunName, getRuntimeMetrics, mapMetricResponse } from '../../utils'
import MetricsTimeline from '../metrics-timeline/MetricsTimeline'
import { sortByPropertyName } from '../../../utils';
import InfoTwoToneIcon from '@mui/icons-material/InfoTwoTone';
import './JobRunTimeline.css';

interface Props {
  isOpen: boolean,
  runsResponse: RunResponse | undefined,
  runs: any;
  tasks: Task[] | undefined,
  jobName: string,
  onClose: () => void;
}

export interface UiMetric {
  name: string,
  from: number,
  to: number,
  time: number,
  category: string,
  stackId: string,
  runId: string,
  id: string,
  fill: string,
  processed?: boolean,
}

const JobRunTimeline = ({ isOpen, runsResponse, runs, jobName, tasks, onClose }: Props) => {
  const [metrics, setMetrics] = useState<any[]>([]);
  const [loading, setIsLoading] = useState<boolean>(true);
  const [names, setNames] = useState<any[]>([]);
  const [responses, setResponses] = useState<any[]>([]);
  const [minDate, setMinDate] = useState<Date>();
  const [maxDate, setMaxDate] = useState<Date>();
  const [workers, setWorkers] = useState<string[] | undefined>([]);
  const [transformedData, setTransformedData] = useState<any>();

  const isOpenReference = useRef<boolean>();
  isOpenReference.current = isOpen;

  useEffect(() => {
    if (!metrics.length) { return; }
    const sortedRunTimeMetrics = sortByPropertyName(metrics, 'order');

    if (isOpenReference.current) {
      fetchMetrics(sortedRunTimeMetrics, names);
    }
  }, [metrics, names, isOpenReference.current]);

  useEffect(() => {
    if (!Object.keys(responses)?.length) { return; }
    const data: any = {}

    Object.keys(responses).map((key: any) => (
      data[key] = responses[key].filter((item: any) => item.key.includes('runtime'))
    ));

    setIsLoading(false);
    setTransformedData(data);
  }, [responses])

  useEffect(() => {
    if (!tasks && !runsResponse) { return; }

    const runs = getLatestRuns(runsResponse, tasks);
    const max = Math.max(...(runs || [])?.map(run => run.info.end_time));
    const min = Math.min(...(runs || [])?.map(run => run.info.start_time));
    setMinDate(new Date(min - 60000));
    setMaxDate(new Date(max + 60000));
    setWorkers(runs?.map(run => getRunName(run)));

    const mappedData = getRuntimeMetrics(runs);
    setMetrics(mappedData.metrics);
    setNames(mappedData.names);
    setIsLoading(true);
  }, [tasks, runsResponse]);

  const fetchMetrics = async (metrics: UiMetric[], names: string[]) => {
    const fetchedResponses: any = names.reduce((acc, name) => ({ ...acc, [name]: []}), {});
    const apiClient = new ApiClient('mlflow/metrics/get-history', true);

    for (const metric of metrics) {
      if (!isOpenReference.current) {
        return;
      }

      const { name, runId, category } = metric;
        try {
          const response = await apiClient.getAll({ params: { metric_key: name, run_uuid: runId, }})
            .then(res => mapMetricResponse(res, fetchedResponses, metric));
          fetchedResponses[category] = [...fetchedResponses[category], ...(response as unknown as any)];

          setResponses({ ...fetchedResponses });
        } catch (error) {
          console.error('Error fetching metrics:', error);
        }
    }


  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      size="full"
    >
      <ModalOverlay />

      <ModalContent className="job-run-timeline" height="100%">
        <ModalHeader display="flex" gap="20px" alignItems="center" justifyContent="center">
          <Text>{jobName} runtime metrics</Text>

          <Popover>
            <PopoverTrigger>
              <InfoTwoToneIcon cursor="pointer" />
            </PopoverTrigger>

            <PopoverContent>
              <PopoverBody display="flex" flexDirection="column" gap="10px" zIndex="10000">
                <Text fontSize="12px" textAlign="left">shift + mousewheel = move timeline left/right</Text>
                <Text fontSize="12px" textAlign="left">alt + mousewheel = zoom in/out</Text>
                <Text fontSize="12px" textAlign="left">ctrl + mousewheel = zoom in/out 10× faster</Text>
                <Text fontSize="12px" textAlign="left">meta + mousewheel = zoom in/out 3x faster (win or cmd + mousewheel)</Text>
              </PopoverBody>
            </PopoverContent>
          </Popover>
        </ModalHeader>

        <ModalCloseButton />

        <ModalBody pb={6} display="flex" height="100%" overflow="hidden" overflowY="auto" flexDirection="column" gap="10px">
          {
            loading ?
            <Box height="100%" display="flex" justifyContent="center" alignItems="center">
              <Loading message={'Fetching data...'} />
            </Box> :
            isOpen && <MetricsTimeline workers={workers} data={transformedData} minDate={minDate} maxDate={maxDate} isOpen={isOpenReference.current} runs={runs}/>
          }
        </ModalBody>
      </ModalContent>
    </Modal>
  )
}

export default JobRunTimeline