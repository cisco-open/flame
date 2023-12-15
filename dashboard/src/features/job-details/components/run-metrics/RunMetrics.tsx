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

import { Box } from '@chakra-ui/react';
import { useEffect, useState } from 'react';
import { Metric, Run } from '../../../../entities/JobDetails';
import useMetrics from '../../hooks/useMetrics';
import { sortMetrics } from '../../utils';
import MetricChart from '../metric-chart/MetricChart';
import './RunMetrics.css';
interface Props {
  run: Run;
}


const RunMetrics = ({ run }: Props) => {
  const [ selectedMetric, setSelectedMetric ] = useState<any>();
  const [ sortedMetrics, setSortedMetrics ] = useState<any>();
  const { data } = useMetrics({ run_uuid: selectedMetric?.run_uuid, metric_key: selectedMetric?.metric_key });

  useEffect(() => {
    if (!run?.data?.metrics) { return; }
    setSortedMetrics([...sortMetrics(run?.data?.metrics)]);
  }, [run]);

  const onMetricSelect = (param: any) => {
    setSelectedMetric({ run_uuid: run.info.run_uuid, metric_key: param.key })
  }

  return (
    <Box className="metrics-container">
      <Box className="metrics-list">
        {
          sortedMetrics?.map((metric: Metric) =>
          <Box className="metric" key={metric.key} onClick={() => onMetricSelect(metric)}>
            <p>{metric.key}</p>
          </Box>
          )
        }
      </Box>

      <Box className="chart-container">
        <MetricChart data={data}/>
      </Box>
    </Box>
  )
}

export default RunMetrics