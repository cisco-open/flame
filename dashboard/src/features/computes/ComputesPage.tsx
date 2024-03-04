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

import { Button, Box, Text, useDisclosure } from '@chakra-ui/react';
import AddIcon from '@mui/icons-material/Add';
import ComputeFormModal from './compute-form-modal/ComputeFormModal';
import ComputesList from './computes-list/ComputesList';
import useComputes from './hooks/useComputes';
import { ComputeFormData } from './types';

const ComputesPage = () => {
  const { isOpen, onOpen, onClose } = useDisclosure();
  const { data, createMutation } = useComputes(onClose);

  const handleClose = () => {
    onClose();
  };

  const onSave = (data: ComputeFormData) => {
    createMutation.mutate({ ...data });
  }

  return (
    <Box gap={5} display="flex" flexDirection="column" height="100%" overflow="hidden">
      <Box display="flex" alignItems="center" justifyContent="space-between" zIndex="1">
        <Text as="h1" fontWeight="bold">COMPUTES</Text>

        <Button leftIcon={<AddIcon fontSize="small" />} alignSelf="flex-end" size="xs" colorScheme="primary">Create New</Button>
      </Box>

      <ComputesList />

      { isOpen && <ComputeFormModal isOpen={isOpen} onClose={handleClose} onSave={onSave} /> }
    </Box>
  )
}

export default ComputesPage