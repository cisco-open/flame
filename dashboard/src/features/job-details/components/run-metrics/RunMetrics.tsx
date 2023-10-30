import { Box } from '@chakra-ui/react';
import { useEffect, useState } from 'react';
import { Metric, Run } from '../../../../entities/JobDetails';
import useMetrics from '../../hooks/useMetrics';
import { sortMetrics } from '../../utils';
import MetricChart from '../metric-chart/MetricChart';
import './RunMetrics.css';
interface Props {
  metrics: Metric[] | undefined;
  run: Run;
}


const RunMetrics = ({ metrics, run }: Props) => {
  const [ selectedMetric, setSelectedMetric ] = useState<any>();
  const [ sortedMetrics, setSortedMetrics ] = useState<any>();
  const { data } = useMetrics({ run_uuid: selectedMetric?.run_uuid, metric_key: selectedMetric?.metric_key });

  useEffect(() => {
    setSortedMetrics(sortMetrics(metrics));
  }, [metrics]);

  const onMetricSelect = (param: any) => {
    setSelectedMetric({ run_uuid: run.info.run_uuid, metric_key: param.key })
  }

  return (
    <Box className="metrics-container">
      <Box className="metrics-list">
        {
          sortedMetrics?.map((param: Metric) =>
          <Box className="metric" key={param.key} onClick={() => onMetricSelect(param)}>
            <p>{param.key}</p>
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