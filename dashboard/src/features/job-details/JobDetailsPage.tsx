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

import { Box, Button, Text, useDisclosure, Popover, PopoverBody, PopoverContent, PopoverTrigger } from "@chakra-ui/react";
import { useNavigate, useParams } from "react-router-dom";
import JobTopology from "./components/job-topology/JobTopology";
import ArrowBackIosIcon from '@mui/icons-material/ArrowBackIos';
import JobRunTimeline from "./components/job-run-timeline/JobRunTimeline";
import useRuns from "./hooks/useRuns";
import useTasks from "../jobs/hooks/useTasks";
import useExperiment from "./hooks/useExperiment";
import useJob from "../jobs/hooks/useJob";
import AssessmentTwoToneIcon from '@mui/icons-material/AssessmentTwoTone';
import React, { useEffect, useRef, useState } from "react";
import { getRunName } from "./utils";
import { FaEllipsisVertical } from "react-icons/fa6";
import EditOutlinedIcon from '@mui/icons-material/EditOutlined';
import PlayCircleOutlineIcon from '@mui/icons-material/PlayCircleOutline';
import StopCircleIcon from '@mui/icons-material/StopCircle';
import useJobs from "../jobs/hooks/useJobs";
import { Job } from "../../entities/Job";
import JobFormModal from "../jobs/components/job-form-modal/JobFormModal";
import ConfirmationDialog from "../../components/confirmation-dialog/ConfirmationDialog";
import DeleteOutlineOutlinedIcon from '@mui/icons-material/DeleteOutlineOutlined';
import useJobStatus from "../jobs/hooks/useJobStatus";
import './JobDetailsPage.css';

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
  const { isOpen: isJobTimelineOpen, onOpen: onJobTimelineOpen, onClose: onJobTimelineClose } = useDisclosure();
  const { isOpen: isEditJobOpen, onOpen: onEditJobOpen, onClose: onEditJobClose } = useDisclosure();
  const { isOpen: isConfirmationOpen, onOpen: onConfirmationOpen, onClose: onConfirmationClosed } = useDisclosure();
  const { data: job } = useJob(id || '');
  const [ runs, setRuns ] = useState<any[]>();
  const [ jobId, setJobId ] = useState('');
  const { updateStatusMutation, deleteMutation } = useJobs(id, undefined, navigate);
  const { data: jobStatus} = useJobStatus(id);
  const { isOpen: isPopoverOpen, onToggle: onPopoverToggle, onClose: onPopoverClose } = useDisclosure();

  const isOpenReference = useRef<boolean>();
  isOpenReference.current = isPopoverOpen;

  useEffect(() => {
    window.addEventListener('click', closeMenuOnClick);

    return () => {
      window.removeEventListener('click', closeMenuOnClick);
    }
  }, [])

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
  }, [runsResponse]);

  const closeMenuOnClick = () => {
    if(isOpenReference.current) {
      onPopoverClose();
    }
  }

  const onDelete = () => {
    deleteMutation.mutate();
  }

  const handleConfirmationClose = () => {
    onConfirmationClosed();
    setJobId('');
  }

  const onStartClick = (event: React.MouseEvent, job: Job) => {
    updateStatusMutation.mutate({
      id: job.id,
      state: 'starting'
    })
  }

  const onStopClick = (event: React.MouseEvent, job: Job) => {
    updateStatusMutation.mutate({
      id: job.id,
      state: 'stopping'
    })
  }

  const openConfirmationModal = (event: React.MouseEvent) => {
    onConfirmationOpen();
  }

  return (
    <Box display="flex" overflow="hidden" flexDirection="column" gap="10px" height="100%" className="job-details">
      <Box display="flex" alignItems="center" position="relative" zIndex="1">
        <Button marginTop="2px" leftIcon={<ArrowBackIosIcon fontSize="small" />} onClick={() => navigate('/jobs?stateFilter=all')} colorScheme="primary" variant='link' size="xs">Jobs</Button>

        <Text as="h2" flex="1" textAlign="center" fontWeight="bold">{ job?.name }</Text>
      </Box>

      <Box width="100%" height="100%" position="relative" bgColor="white" borderRadius="10px">
        <Box position="absolute" zIndex="1" top="5px" right="5px" display="flex" flexDirection="column" alignItems="flex-end" gap="20px">
          <Popover isOpen={isPopoverOpen}>
            <PopoverTrigger>
              <Button onClick={(event: React.MouseEvent) => { event.stopPropagation(); onPopoverToggle() }} className='menu-button' leftIcon={<FaEllipsisVertical />} />
            </PopoverTrigger>

            <PopoverContent>
              <PopoverBody className='job-details-popover-body' onClick={onPopoverClose}>
                <Box className="job-details-menu-item" onClick={onEditJobOpen}>
                  <EditOutlinedIcon fontSize="small"/>

                  Edit
                </Box>

                <Box
                  className="design-details-menu-item"
                  onClick={(event) => jobStatus?.state !== 'running' ? onStartClick(event, job) : onStopClick(event, job)}
                >
                  {
                    jobStatus?.state !== 'running' ?
                      <PlayCircleOutlineIcon  fontSize="small"/> :
                      <StopCircleIcon fontSize="small"/>
                  }

                  {jobStatus?.state !== 'running' ? 'Start Job' : 'Stop Job' }
                </Box>

                <Box className="design-details-menu-item" onClick={onJobTimelineOpen}>
                  <AssessmentTwoToneIcon fontSize="small"/>

                  Runtime Metrics
                </Box>

                <Box className="design-details-menu-item" onClick={(event) => openConfirmationModal(event)}>
                  <DeleteOutlineOutlinedIcon fontSize="small"/>

                  Delete Job
                </Box>
              </PopoverBody>
            </PopoverContent>
          </Popover>
        </Box>

        <JobTopology
          tasks={tasks}
          experiment={experiment}
          mutate={mutate}
          runs={runs}
        />
      </Box>

      { 
        isJobTimelineOpen &&
        <JobRunTimeline
          isOpen={isJobTimelineOpen}
          onClose={onJobTimelineClose}
          runsResponse={runsResponse}
          tasks={tasks}
          jobName={job?.name}
          runs={runs}
        />
      }

      { isEditJobOpen && <JobFormModal isOpen={isEditJobOpen} job={job} onClose={onEditJobClose}/> }

      {
        isConfirmationOpen &&
        <ConfirmationDialog
          actionButtonLabel={'Delete'}
          message={'Are sure you want to delete this job?'}
          buttonColorScheme={'red'}
          isOpen={isConfirmationOpen}
          onClose={handleConfirmationClose}
          onAction={onDelete}
        />
      }
    </Box>
  )
}

export default JobDetailsPage;

