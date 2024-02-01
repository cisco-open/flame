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

import { Box, Button, Text, useDisclosure, Icon } from "@chakra-ui/react";
import { useNavigate, useParams } from "react-router-dom";
import JobTopology from "./components/job-topology/JobTopology";
import ArrowBackIosIcon from '@mui/icons-material/ArrowBackIos';
import JobRunTimeline from "./components/job-run-timeline/JobRunTimeline";
import useRuns from "./hooks/useRuns";
import useTasks from "../jobs/hooks/useTasks";
import useExperiment from "./hooks/useExperiment";
import useJob from "../jobs/hooks/useJob";
import AssessmentTwoToneIcon from '@mui/icons-material/AssessmentTwoTone';
import { useEffect, useState } from "react";
import { getRunName } from "./utils";

export const fitViewOptions = {
  padding: 1,
  maxZoom: 4
}

const JobDetailsPage = () => {
  const navigate = useNavigate();
  const { data: runsResponse, mutate } = useRuns();
  const { id } = useParams();
  const { data: tasks } = useTasks(id || '');
  const { data: experiment } = useExperiment(id || '');
  const { isOpen, onOpen, onClose } = useDisclosure();
  const { data: job } = useJob(id || '');
  const [ runs, setRuns ] = useState<any[]>();

  useEffect(() => {
    if (!runsResponse?.runs) { return; }
    const runs = runsResponse?.runs.map(run => {
      const runName = getRunName(run);
      const runNameSlices = runName.split('-');
      const taskId = runNameSlices[runNameSlices.length - 1];
      return {
        ...run,
        taskId,
      }
    });
    setRuns(runs);
  }, [runsResponse])

  return (
    <Box display="flex" flexDirection="column" gap="10px" height="100%">
      <Box display="flex" alignItems="center" position="relative" zIndex="1">
        <Button marginTop="2px" leftIcon={<ArrowBackIosIcon fontSize="small" />} onClick={() => navigate('/jobs')} colorScheme="primary" variant='link' size="xs">Back</Button>

        <Text as="h2" flex="1" textAlign="center" fontWeight="bold">{ job?.name }</Text>
      </Box>

      <Box width="100%" height="100%" position="relative" bgColor="white" borderRadius="10px">
        <Icon cursor="pointer" position="absolute" top="10px" right="10px" width="40px" height="40px" zIndex="1" as={AssessmentTwoToneIcon} onClick={onOpen} />

        <JobTopology
          tasks={tasks}
          experiment={experiment}
          mutate={mutate}
          runs={runs}
        />
      </Box>

      <JobRunTimeline
        isOpen={isOpen}
        onClose={onClose}
        runsResponse={runsResponse}
        tasks={tasks}
        jobName={job?.name}
        runs={runs}
      />
    </Box>
  )
}

export default JobDetailsPage;

