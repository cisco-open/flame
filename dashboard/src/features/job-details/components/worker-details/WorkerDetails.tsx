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

import { TabList, TabPanel, Tab, TabPanels, Tabs } from '@chakra-ui/react';
import { useContext, useState } from 'react';
import { JobDetailsContext } from '../../JobDetailsContext';
import RunMetrics from '../run-metrics/RunMetrics';
import RunModelArtefact from '../run-model-artefact/RunModelArtefact';
import RunParameters from '../run-parameters/RunParameters';
import './WorkerDetails.css';

interface Props {
  runDetails: any;
}

const WorkerDetails = ({ runDetails }: Props) => {
  const { artifact } = useContext(JobDetailsContext);
  return (
    <Tabs className="run-details-tabs">
      <TabList>
        <Tab fontSize="12px">Hyperparameters</Tab>
        <Tab fontSize="12px">Metrics</Tab>
        <Tab fontSize="12px">Model Artifact</Tab>
      </TabList>

      <TabPanels className="run-details-tab-panels">
        <TabPanel className="run-details-tab-panel">
          <RunParameters parameters={runDetails?.data?.params} />
        </TabPanel>
        <TabPanel className="run-details-tab-panel">
          <RunMetrics run={runDetails}/>
        </TabPanel>
        <TabPanel className="run-details-tab-panel">
          <RunModelArtefact runDetails={runDetails} />
        </TabPanel>
      </TabPanels>
    </Tabs>
  )
}

export default WorkerDetails