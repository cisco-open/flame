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

import { TableContainer, Table, Thead, Tr, Th, Tbody, Td, Text, Box, Tooltip, useDisclosure, IconButton, Menu, MenuButton, MenuItem, MenuList, Icon } from "@chakra-ui/react";
import { Job } from "../../../entities/Job";
import PlayCircleOutlineIcon from '@mui/icons-material/PlayCircleOutline';
import StopCircleIcon from '@mui/icons-material/StopCircle';
import useJobs from "../hooks/useJobs";
import { useNavigate } from "react-router-dom";
import React, { useState } from "react";
import EditOutlinedIcon from '@mui/icons-material/EditOutlined';
import DeleteOutlineOutlinedIcon from '@mui/icons-material/DeleteOutlineOutlined';
import ConfirmationDialog from "../../../components/confirmation-dialog/ConfirmationDialog";
import { FaEllipsisVertical } from "react-icons/fa6";
import { colors } from "../../..";

const columns = ['Name', 'State', ''];

interface Props {
  openJobModal: (job: Job) => void;
}

const JobsList = ({ openJobModal }: Props) => {
  const [jobId, setJobId] = useState('');
  const navigate = useNavigate();
  const { data: jobs, updateStatusMutation, deleteMutation } = useJobs(jobId);
  const { isOpen, onOpen, onClose } = useDisclosure();

  const onStartClick = (event: React.MouseEvent, job: Job) => {
    event.stopPropagation();
    setJobId(job.id);

    updateStatusMutation.mutate({
      id: job.id,
      state: 'starting'
    })
  }

  const onStopClick = (event: React.MouseEvent, job: Job) => {
    event.stopPropagation();
    setJobId(job.id);

    updateStatusMutation.mutate({
      id: job.id,
      state: 'stopping'
    })
  }

  const openConfirmationModal = (event: React.MouseEvent, jobId: string) => {
    event.stopPropagation();
    setJobId(jobId);

    onOpen();
  }

  const onDelete = () => {
    deleteMutation.mutate();
    onClose();
  }

  const handleConfirmationClose = () => {
    setJobId('');
    onClose();
  }

  const goToJobDetails = (job: Job) => {
    navigate(`/jobs/${job.id}`)
  }

  const onEditClick = (event: React.MouseEvent, job: Job) => {
    event.stopPropagation();

    openJobModal(job);
  }

  return (
    // <SimpleGrid columns={3} spacing="20px">
    //   {jobs?.map(sjob => 
    //     <JobCard />
    //   )}
    // </SimpleGrid>
    <TableContainer flex={1} overflowY="auto" zIndex="1" backgroundColor="white" borderRadius="10px" padding="10px">
        <Table variant='simple' fontSize={12} size="sm">
        <Thead>
            <Tr>
                {columns.map(column => <Th key={column}>{column}</Th>)}
            </Tr>
        </Thead>

        <Tbody>
            {jobs?.map((job: Job) =>
            <Tr height="50px" key={job.id}>
                <Td>
                  <Text as="span" cursor="pointer" textDecoration="underline" color={colors.primary.normal} onClick={() => goToJobDetails(job)}>{job.name}</Text>
                </Td>

                <Td>{job.state}</Td>

                <Td>
                <Box display="flex" gap="10px" justifyContent="flex-end">
                  <Menu>
                    <MenuButton
                      as={IconButton}
                      aria-label='Options'
                      icon={<Icon as={FaEllipsisVertical} />}
                      variant='outline'
                      onClick={(event) => event.stopPropagation()}
                      border="none"
                    />
                    <MenuList>
                      <MenuItem
                        onClick={(event) => onEditClick(event, job)}
                        icon={<EditOutlinedIcon fontSize="small"/>}
                      >
                        Edit
                      </MenuItem>

                      <MenuItem
                        onClick={(event) => job.state !== 'running' ? onStartClick(event, job) : onStopClick(event, job)}
                        icon={
                          job.state !== 'running' ?
                            <PlayCircleOutlineIcon  fontSize="small"/> :
                            <StopCircleIcon fontSize="small"/>
                        }
                      >
                        {job.state !== 'running' ? 'Start Job' : 'Stop Job' }
                      </MenuItem>

                      <MenuItem
                        onClick={(event) => openConfirmationModal(event, job.id)}
                        icon={<DeleteOutlineOutlinedIcon fontSize="small"/>}
                      >
                        Delete
                      </MenuItem>
                    </MenuList>
                  </Menu>
                </Box>
                </Td>
            </Tr>
            )}
        </Tbody>
        </Table>

        <ConfirmationDialog
          actionButtonLabel={'Delete'}
          message={'Are sure you want to delete this job?'}
          buttonColorScheme={'red'}
          isOpen={isOpen}
          onClose={handleConfirmationClose}
          onAction={onDelete}
        />
    </TableContainer>
  )
}

export default JobsList;