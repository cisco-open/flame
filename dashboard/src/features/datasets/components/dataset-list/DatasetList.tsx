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
import { DesignForm } from '../../../../entities/DesignForm';
import AddIcon from '@mui/icons-material/Add';
import { Dataset } from '../../../../entities/Dataset';
import DatasetTable from '../dataset-table/DatasetTable';
import useDatasets from '../../../../hooks/useDatasets';
import DatasetForm from '../dataset-form-modal/DatasetFormModal';
import { useState } from 'react';
import { DEFAULT_DATA_FORMAT, DEFAULT_REALM } from '../../constants';
import { LOGGEDIN_USER } from '../../../../constants';

const DatasetList = () => {
  const { isOpen, onOpen, onClose } = useDisclosure();
  const [ isSaveSuccess, setIsSaveSuccess ] = useState<boolean>(false);
  const { data: datasets, createMutation } = useDatasets({ onClose, setIsSaveSuccess });

  const handleEdit = (dataset: Dataset) => {

  }

  const handleClose = () => {
    onClose();
  }

  const handleSave = (data: any) => {
    createMutation.mutate({ ...data, realm: DEFAULT_REALM, dataFormat: DEFAULT_DATA_FORMAT, userId: LOGGEDIN_USER.name });
  }

  return (
    <Box gap={5} display="flex" flexDirection="column" height="100%" overflow="hidden">
      <Box display="flex" alignItems="center" justifyContent="space-between">
        <Text as="h1" fontWeight="bold">DATASETS</Text>

        <Button leftIcon={<AddIcon fontSize="small" />} onClick={onOpen} alignSelf="flex-end" variant='outline' size="xs" colorScheme="teal">Create New</Button>
      </Box>
      <DatasetTable
        datasets={datasets}
        onEdit={(dataset: Dataset) => handleEdit(dataset)}
      />

      <DatasetForm isOpen={isOpen} isSaveSuccess={isSaveSuccess} onClose={handleClose} onSave={(data: DesignForm) => handleSave(data)} />
    </Box>
  )
}

export default DatasetList