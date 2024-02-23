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

import { Box, Icon, IconButton, Menu, MenuButton, MenuItem, MenuList, Table, TableContainer, Tbody, Td, Th, Thead, Tooltip, Tr, useDisclosure } from '@chakra-ui/react';
import { Dataset } from '../../../../entities/Dataset'
import EditOutlinedIcon from '@mui/icons-material/EditOutlined';
import { useEffect, useState } from 'react';
import { FaEllipsisVertical } from "react-icons/fa6";
import DeleteOutlineOutlinedIcon from '@mui/icons-material/DeleteOutlineOutlined';
import ConfirmationDialog from '../../../../components/confirmation-dialog/ConfirmationDialog';
import { COLORS } from '../../../../constants';
interface Props {
  datasets: Dataset[] | undefined;
  onEdit?: (dataset: Dataset) => void;
  onDelete?: (id: string) => void;
}

const DatasetTable = ({ datasets, onEdit, onDelete }: Props) => {
  const columns = ['Name', 'Realm', 'User ID', 'Data Format', 'Compute ID', ''];
  const [filteredDatasets, setFilteredDatasets] = useState<Dataset[] | undefined>([]);
  const { isOpen, onOpen, onClose } = useDisclosure();
  const [ datasetId, setDatasetId ] = useState('');

  useEffect(() => {
    setFilteredDatasets(datasets);
  }, [datasets])

  const onEditClicked = (event: React.MouseEvent, dataset: Dataset) => {
    event.stopPropagation();

    if (onEdit) {
      onEdit(dataset);
    }
  }

  const onDeleteClicked = (event: React.MouseEvent, id: string) => {
    event?.stopPropagation();
    setDatasetId(id);
    onOpen();
  }

  const handleConfirmationClose = () => {
    onClose();
    setDatasetId('');
  }

  const onDeleteConfirm = () => {
    if (onDelete) {
      onDelete(datasetId);
    }
    setDatasetId('');
    onClose();
  }

  return (
    <TableContainer flex={1} overflowY="auto" zIndex="1" backgroundColor={COLORS.offWhite} borderRadius="10px" padding="10px">
    <Table variant='simple' fontSize="12px" size="sm">
    <Thead>
        <Tr>
            {columns.map(column => <Th key={column}>{column}</Th>)}
        </Tr>
    </Thead>

    <Tbody>
        {filteredDatasets?.map((dataset: Dataset) =>
        <Tr height="50px" key={dataset.id}>
          <Td>{dataset.name}</Td>

          <Td>{dataset.realm}</Td>

          <Td>{dataset.userId}</Td>

          <Td>{dataset.dataFormat}</Td>

          <Td>{dataset.computeId}</Td>

          <Td>
            {
              onEdit && onDelete &&
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
                      onClick={(event) => onEditClicked(event, dataset)}
                      icon={<EditOutlinedIcon fontSize="small"/>}
                    >
                      Edit
                    </MenuItem>

                    <MenuItem
                      onClick={(event) => onDeleteClicked(event, dataset.id)} 
                      icon={<DeleteOutlineOutlinedIcon fontSize="small"/>}
                    >
                      Delete
                    </MenuItem>
                  </MenuList>
                </Menu>
              </Box>
            }
          </Td>
        </Tr>
        )}
    </Tbody>
    </Table>

    <ConfirmationDialog
      actionButtonLabel={'Delete'}
      message={'Are sure you want to delete this dataset?'}
      buttonColorScheme={'red'}
      isOpen={isOpen}
      onClose={handleConfirmationClose}
      onAction={onDeleteConfirm}
    />
</TableContainer>
  )
}

export default DatasetTable