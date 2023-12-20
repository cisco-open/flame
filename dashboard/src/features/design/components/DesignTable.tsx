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

import { Box, Icon, IconButton, Text, Menu, MenuButton, MenuItem, MenuList, Table, TableContainer, Tbody, Td, Th, Thead, Tooltip, Tr, useDisclosure } from '@chakra-ui/react';
import DeleteOutlineOutlinedIcon from '@mui/icons-material/DeleteOutlineOutlined';
import EditOutlinedIcon from '@mui/icons-material/EditOutlined';
import { useNavigate } from 'react-router-dom';
import { Design } from '../../../entities/Design';
import { MouseEvent, useState } from 'react';
import ConfirmationDialog from '../../../components/confirmation-dialog/ConfirmationDialog';
import { FaEllipsisVertical } from "react-icons/fa6";
import { colors } from '../../..';

interface Props {
    designs: Design[] | undefined;
    onDelete: (id: string) => void;
    onEdit: (design: Design) => void;
}

const DesignTable = ({ designs, onDelete, onEdit }: Props) => {
  const columns = ['Name', 'ID', ''];
  const [ designId, setDesignId ] = useState('');
  const navigate = useNavigate();
  const { isOpen, onOpen, onClose } = useDisclosure();

  const goToDesignDetails = (id: string): void => {
    navigate(`/design/${id}`);
  }

  const onEditClicked = (event: MouseEvent, design: Design) => {
    event?.stopPropagation();

    onEdit(design);
  }

  const onDeleteClicked = (event: MouseEvent, id: string) => {
    event?.stopPropagation();
    setDesignId(id);
    onOpen();
  }

  const onDeleteConfirm = () => {
    onDelete(designId);
    setDesignId('');
    onClose();
  }

  const handleConfirmationClose = () => {
    onClose();
    setDesignId('');
  }

  return (
    <TableContainer flex={1} overflowY="auto" zIndex="1" backgroundColor="white" borderRadius="10px" padding="10px">
        <Table variant='simple' fontSize={12} size="sm">
        <Thead>
            <Tr>
                {columns.map(column => <Th key={column}>{column}</Th>)}
            </Tr>
        </Thead>

        <Tbody>
            {designs?.map((design: Design) =>
            <Tr height="50px" key={design.id}>
                <Td padding="10px 20px">
                  <Text onClick={() => goToDesignDetails(design.id)} as="span" cursor="pointer" textDecoration="underline" color={colors.primary.normal}>{design.name}</Text>
                </Td>

                <Td padding="10px 20px">{design.id}</Td>

                <Td padding="10px 20px">
                <Box display="flex" gap="10px" justifyContent="flex-end">
                  <Menu>
                      <MenuButton
                        as={IconButton}
                        aria-label='Options'
                        icon={<Icon as={FaEllipsisVertical} />}
                        variant='outline'
                        onClick={(event) => event.stopPropagation()}
                        size="md"
                        border="none"
                      />
                      <MenuList>
                        <MenuItem
                          onClick={(event) => onEditClicked(event, design)}
                          icon={<EditOutlinedIcon fontSize="small"/>}
                        >
                          Edit
                        </MenuItem>

                        <MenuItem
                          onClick={(event) => onDeleteClicked(event, design.id)} 
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
          message={'Are sure you want to delete this design?'}
          buttonColorScheme={'red'}
          isOpen={isOpen}
          onClose={handleConfirmationClose}
          onAction={onDeleteConfirm}
        />
    </TableContainer>
  )
}

export default DesignTable