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
import { Dataset } from '../../../../entities/Dataset';
import DatasetTable from '../dataset-table/DatasetTable';
import useDatasets from '../../../../hooks/useDatasets';
import { useState } from 'react';
import { DEFAULT_DATA_FORMAT } from '../../constants';
import { LOGGEDIN_USER } from '../../../../constants';
import DatasetFormModal from '../dataset-form-modal/DatasetFormModal';
import { DatasetForm } from '../../types';
import { DatasetsContext } from '../../DatasetsContext';

const DatasetList = () => {
  const { isOpen, onOpen, onClose } = useDisclosure();
  const [ datasetId, setDatasetId ] = useState<string>('');
  const [ datasetInEdit, setDatasetInEdit ] = useState<Dataset>();
  const [ isSaveSuccess, setIsSaveSuccess ] = useState<boolean>(false);
  const { data: datasets, createMutation, deleteMutation, updateMutation } = useDatasets({ onClose, setIsSaveSuccess, datasetId, setDatasetId });

  const handleClose = () => {
    onClose();
  }

  const handleSave = (data: DatasetForm) => {
    datasetInEdit ?
      updateMutation.mutate(data) :
      createMutation.mutate({ ...data, dataFormat: DEFAULT_DATA_FORMAT, userId: LOGGEDIN_USER.name });
  }

  const handleDelete = (id: string) => {
    setDatasetId(id);
    deleteMutation.mutate(id);
  }

  const handleEdit = (dataset: Dataset) => {
    setDatasetInEdit(dataset);
    onOpen();
  }

  return (
    <DatasetsContext.Provider value={{ datasetInEdit }}>
      <Box gap={5} display="flex" flexDirection="column" height="100%" overflow="hidden">
        <Box display="flex" alignItems="center" justifyContent="space-between" zIndex="1">
          <Text as="h1" fontWeight="bold">DATASETS</Text>

          <Button leftIcon={<AddIcon fontSize="small" />} onClick={onOpen} alignSelf="flex-end" size="xs" colorScheme="primary">Create New</Button>
        </Box>
        <DatasetTable
          datasets={datasets}
          onDelete={(id: string) => handleDelete(id)}
          onEdit={(dataset: Dataset) => handleEdit(dataset)}
        />

        <DatasetFormModal isOpen={isOpen} isSaveSuccess={isSaveSuccess} onClose={handleClose} onSave={(data: DatasetForm) => handleSave(data)} />
      </Box>
    </DatasetsContext.Provider>
  )
}

export default DatasetList