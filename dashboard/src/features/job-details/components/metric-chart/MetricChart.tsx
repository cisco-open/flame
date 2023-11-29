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

import { Box, Radio, RadioGroup, Stack, Text } from '@chakra-ui/react';
import { format } from 'date-fns';
import { useEffect, useState } from 'react';
import { LineChart, CartesianGrid, Tooltip, XAxis, YAxis, Legend, Line } from 'recharts';
import { Metric } from '../../../../entities/JobDetails';
import ChartLegend from '../chart-legend/ChartLegend';
import CustomTick from '../custom-tick/CustomTick';
import './MetricChart.css';

interface Props {
  data: any;
}

const MetricChart = ({ data }: Props) => {
  const [ chartData, setChartData ] = useState<any>();
  const [ xAxisValue, setXAxisValue ] = useState<string>('date');
  useEffect(() => {
    if (!data) { return; }

    setChartData(getChartData(data?.metrics));
  }, [data]);

  const getChartData = (data: Metric[]) => {
    const minTimestamp = Math.min.apply(Math, data?.map((d: Metric) => d.timestamp));
    return {
      label: data[0].key,
      data: data?.map((d: any) => ({
        key: d.key,
        value: d.value,
        roundedValue: Math.round(d.value),
        date: `${format(d.timestamp, 'mm/dd/yyyy')} ${format(d.timestamp, 'hh:mm')}`,
        step: d.step,
        relativeDate: Math.round((d.timestamp - minTimestamp) / 1000),
      }))
    }
  }

  if (!chartData?.data) {
    return (
      <Box className="chart-zero-state">
        <Text>Select metric</Text>
      </Box>
    )
  }

  const setChartXAxis = (value: any) => {
    setXAxisValue(value);
  }

  const formatYAxis = (value: number) => {
    if (value.toString().length >= 7) {
      return `${Math.round(+value / 1000000)} M`;
    }

    return `${value}`;
  }

  return (
    <Box className="metric-chart-container">
      <RadioGroup onChange={setChartXAxis} value={xAxisValue}>
        <Stack direction='column'>
          <Radio size='sm' fontSize="12px" value='step'>Step</Radio>
          <Radio size='sm' fontSize="12px" value='date'>Date</Radio>
          <Radio size='sm' fontSize="12px" value='relativeDate'>Relative Date</Radio>
        </Stack>
      </RadioGroup>
      {chartData?.data && <LineChart
          data={chartData.data}
          width={500}
          height={300}
          margin={{
            top: 5,
            right: 30,
            left: 20,
            bottom: 5,
          }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey={xAxisValue} tick={<CustomTick />}>
          </XAxis>
          <YAxis tickFormatter={formatYAxis}/>
          <Tooltip wrapperClassName="metric-tooltip"/>
          <Legend content={<ChartLegend label={chartData.label}/>}/>
          <Line type="monotone" dataKey="value" stroke="#FF0000" activeDot={{ r: 5 }} />
        </LineChart>}
    </Box>
  )
}

export default MetricChart;