/**
 * Copyright 2024 Cisco Systems, Inc. and its affiliates
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
import { Job } from '../../../../entities/Job';
import './DashboardJobsChart.css';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
import { Pie } from 'react-chartjs-2';
import { JOB_STATE_COLOR } from '../../constants';

ChartJS.register(ArcElement, Tooltip, Legend);

interface ChartData {
  labels: string[],
  datasets: {
    label: string,
    data: number[],
    backgroundColor: string[],
    borderColor: string[],
    borderWidth: number
  }[]
}

interface Props {
  jobs: Job[] | undefined;
}

const DashboardJobsChart = ({ jobs }: Props) => {
  const [ chartData, setChartData ] = useState<ChartData | undefined>();

  useEffect(() => {
    const jobState: { [key: string]: { count: number, color: string, borderColor: string }} = {};
    const data: ChartData = {
      labels: [],
      datasets: [{
        label: '',
        data: [],
        backgroundColor: [],
        borderColor: [],
        borderWidth: 1,
      }]
    }
    jobs?.forEach((item) => {
      if (!jobState[item.state]) {
        jobState[item.state] = {
          count: 1,
          color: JOB_STATE_COLOR[item.state].color,
          borderColor: JOB_STATE_COLOR[item.state].borderColor,
        };

        return;
      }

      jobState[item.state] = { ...jobState[item.state], count: jobState[item.state].count + 1 };
    });

    data.labels = Object.keys(jobState);
    data.datasets = [{
      label: 'Job status count: ',
      data: Object.keys(jobState).map(key => jobState[key].count),
      backgroundColor: Object.keys(jobState).map(key => jobState[key].color),
      borderColor: Object.keys(jobState).map(key => jobState[key].color),
      borderWidth: 1,
    }];

    setChartData(data);
  }, [jobs])

  return (
    <>
      { chartData && <Pie data={chartData as unknown as any} options={{ plugins: { legend: { display: false } } } } /> }
    </>
  )
}

export default DashboardJobsChart