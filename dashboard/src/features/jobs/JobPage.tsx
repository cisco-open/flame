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

import { Box, Button, useDisclosure, Text } from '@chakra-ui/react';
import AddIcon from '@mui/icons-material/Add';
import { useEffect, useState } from 'react';
import { Job } from '../../entities/Job';
import JobFormModal from './components/job-form-modal/JobFormModal';
import JobsList from './components/JobsList';
import useJob from './hooks/useJob';

const JobPage = () => {
  const { isOpen, onOpen, onClose } = useDisclosure();
  const [ jobInEdit, setJobInEdit ] = useState<Job | null>(null);
  const { data: job } = useJob(jobInEdit?.id || '');

  useEffect(() => {
    if (job) {
      onOpen();
    }
  }, [job]);

  const handleClose = () => {
    setJobInEdit(null);
    onClose();
  };

  const handleEditJob = (job: Job) => {
    setJobInEdit(job);
  }

  return (
    <Box gap={5} display="flex" flexDirection="column" height="100%" overflow="hidden">
      <Box display="flex" alignItems="center" justifyContent="space-between" zIndex="1">
        <Text as="h1" fontWeight="bold">JOBS</Text>

        <Button leftIcon={<AddIcon fontSize="small" />} onClick={onOpen} alignSelf="flex-end" size="xs" colorScheme="primary">Create New</Button>
      </Box>

      <JobsList openJobModal={(job: Job) => handleEditJob(job)} />

      { isOpen && <JobFormModal isOpen={isOpen} job={job} onClose={handleClose}/> }
    </Box>
  )
}

export default JobPage;