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

import { Badge, Box, Text } from '@chakra-ui/layout';
import { useNavigate } from 'react-router';
import useDatasets from '../../hooks/useDatasets';
import useDesigns from '../design/hooks/useDesigns';
import useJobs from '../jobs/hooks/useJobs';
import './DashboardPage.css';
import DashboardList from './components/dashboard-list/DashboardList';
import DashboardJobsChart from './components/dashboard-jobs-chart/DashboardJobsChart';
import WorkOutlineOutlinedIcon from '@mui/icons-material/WorkOutlineOutlined';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import { useEffect, useState } from 'react';
import { mapJobsToStatus } from './utils';
import { Link } from 'react-router-dom';
import DesignServicesOutlinedIcon from '@mui/icons-material/DesignServicesOutlined';
import StorageIcon from '@mui/icons-material/Storage';
import { Design } from '../../entities/Design';
import DashboardDesignCard from './components/dashboard-design-card/DashboardDesignCard';
import { SimpleGrid } from '@chakra-ui/react';
import DatasetTable from '../datasets/components/dataset-table/DatasetTable';
import { JOB_STATE_COLOR } from './constants';
import { JOB_STATE } from '../../constants';

export enum DashboardEntityType {
  Design,
  Dataset,
  Job
}

const DashboardPage = () => {
  const { data: jobs } = useJobs();
  const { data: designs } = useDesigns();
  const { data: datasets } = useDatasets();
  const [ jobsState, setJobsState ] = useState<{ [key: string]: number }>();
  const [ mappedDesigns, setMappedDesigns ] = useState<Design[]>();

  useEffect(() => {
    if (!designs?.length) { return; }

    setMappedDesigns([...designs.slice(0, 6)])
  }, [designs])

  useEffect(() => {
    if (!jobs) { return; }

    setJobsState(mapJobsToStatus(jobs));
  }, [jobs]);

  return (
    <Box className="dashboard-page-container">
      {/* jobs */}
      <Box className="dashboard-page-container__item">
        <Text as="h3" className="dashboard-page-container__item__title">
          <WorkOutlineOutlinedIcon sx={{ color: 'gray' }} fontSize="inherit"/>

          <Link to={
            {
              pathname: '/jobs',
              search: `?stateFilter=all`,
            }}
            className="flame-link"
          >
            JOBS
          </Link>
        </Text>

        <Box className="dashboard-page-container__item__content">
          <Box display="flex" flexDirection="column" justifyContent="space-between" borderRight="1px solid gray">
            <Box className="dashboard-page-container__item__jobs-count">
              {
                jobsState && Object.keys(jobsState).map(state => 
                  <Box key={state} className="dashboard-page-container__item__jobs-count__item">
                    <Link to={{
                      pathname: '/jobs',
                      search: `?stateFilter=${state}`,
                    }}
                    className="flame-link">
                      {
                        state === JOB_STATE.failed ?
                          <ErrorIcon sx={{ color: JOB_STATE_COLOR[state].borderColor }} fontSize="inherit"/> :
                          <CheckCircleIcon sx={{ color: JOB_STATE_COLOR[state].borderColor }} fontSize="inherit"/>
                      }

                      { state }
                    </Link>

                    <Text as="p"> - { jobsState?.[state] }</Text>
                  </Box>
                )
              }
            </Box>

            <Box className="dashboard-page-container__item__jobs-chart">
              <DashboardJobsChart jobs={jobs} />
            </Box>
          </Box>

          <Box className="dashboard-page-container__item__jobs-list">
            <DashboardList data={jobs?.slice(0, 6)} />
          </Box>
        </Box>
      </Box>

      {/* designs */}
      <Box className="dashboard-page-container__designs-datasets__item">
        <Text as="h3" className="dashboard-page-container__item__title flame-link">
          <DesignServicesOutlinedIcon sx={{ color: 'gray' }} fontSize="inherit"/>

          <Link to="/design">DESIGNS</Link>
        </Text>
        <SimpleGrid columns={3} spacing={5}>
          {
            mappedDesigns?.map((design, index) =>
            <div onMouseDown={(event) => { event.preventDefault(); event.stopPropagation(); }} key={`${design.id}-${index}`}>
              <DashboardDesignCard design={design} />
            </div>)
          }
        </SimpleGrid>
      </Box>

      {/* datasets */}
      <Box className="dashboard-page-container__designs-datasets__item">
        <Text as="h3" className="dashboard-page-container__item__title flame-link">
          <StorageIcon sx={{ color: 'gray' }} fontSize="inherit"/>

          <Link to="/datasets">DATASETS</Link>
        </Text>

        <Box className="grid-item">
          <DatasetTable datasets={datasets?.slice(0, 6)}/>
        </Box>
      </Box>
    </Box>
  )
}

export default DashboardPage