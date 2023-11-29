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

import { Box, Table, TableContainer, Tbody, Td, Th, Thead, Tooltip, Tr } from '@chakra-ui/react';
import { Dataset } from '../../../../entities/Dataset'
import EditOutlinedIcon from '@mui/icons-material/EditOutlined';
import { useEffect, useState } from 'react';

interface Props {
  datasets: Dataset[] | undefined;
  onEdit: (dataset: Dataset) => void;
}

const DatasetTable = ({ datasets, onEdit }: Props) => {
  const columns = ['Name', 'Realm', 'User ID', 'Data Format', 'Compute ID', ''];
  const [filteredDatasets, setFilteredDatasets] = useState<Dataset[] | undefined>([]);

  useEffect(() => {
    setFilteredDatasets(datasets);
  }, [datasets])

  const onEditClicked = (event: any, dataset: Dataset) => {
    onEdit(dataset);
  }

  return (
    <TableContainer flex={1} overflowY="auto">
    <Table variant='simple' fontSize="12px" size="sm">
    <Thead>
        <Tr>
            {columns.map(column => <Th key={column}>{column}</Th>)}
        </Tr>
    </Thead>

    <Tbody>
        {filteredDatasets?.map((dataset: Dataset) =>
        <Tr height="50px" key={dataset.id} cursor="pointer">
            <Td>{dataset.name}</Td>

            <Td>{dataset.realm}</Td>

            <Td>{dataset.userId}</Td>

            <Td>{dataset.dataFormat}</Td>

            <Td>{dataset.computeId}</Td>

            <Td>
            <Box display="flex" gap="10px" justifyContent="flex-end">
                <Tooltip label="Edit" fontSize="inherit">
                    <EditOutlinedIcon onClick={(event) => onEditClicked(event, dataset)} cursor="pointer" fontSize="small"/>
                </Tooltip>
            </Box>
            </Td>
        </Tr>
        )}
    </Tbody>
    </Table>
</TableContainer>
  )
}

export default DatasetTable