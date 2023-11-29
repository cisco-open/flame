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


import AddIcon from '@mui/icons-material/Add';
import { Box, Button, useDisclosure, Text } from '@chakra-ui/react';
import DesignTable from './DesignTable';
import DesignFormModal from './DesignFormModal';
import { Design } from '../../../entities/Design';
import useDesigns from '../hooks/useDesigns';
import { useEffect, useState } from 'react';
import { DesignContext } from '../DesignContext';
import { DesignForm } from '../../../entities/DesignForm';
import { LOGGEDIN_USER } from '../../../constants';

const DesignList = () => {
  const defaultDesignInEdit = {
    description: '',
    name: '',
    id: '',
    userId: ''
  };

  const { data, isLoading, createMutation, deleteMutation } = useDesigns();
  const { isOpen, onOpen, onClose } = useDisclosure();
  const [ designInEdit, setDesignInEdit ] = useState<Design>(defaultDesignInEdit);

  useEffect(() => {
    handleClose();
  }, [createMutation.isSuccess])

  const handleSave = (data: DesignForm) => {
    createMutation.mutate({ ...data, userId: LOGGEDIN_USER.name });
  };

  const handleDelete = (id: string) => {
    deleteMutation.mutate(id);
  }

  const handleEdit = (design: Design) => {
    setDesignInEdit(design);
    onOpen();
  }

  const handleClose = () => {
    onClose();
    setDesignInEdit(defaultDesignInEdit);
  }

  if (isLoading) {
    return <p>Loading...</p>
  }

  return (
    <DesignContext.Provider value={{ designInEdit }}>
      <Box gap="20px" display="flex" flexDirection="column" height="100%" overflow="hidden">
        <Box display="flex" alignItems="center" justifyContent="space-between">
          <Text as="h1" fontWeight="bold">DESIGNS</Text>
          
          <Button leftIcon={<AddIcon fontSize="small" />} onClick={onOpen} alignSelf="flex-end" variant='outline' size="xs" colorScheme="teal">Create New</Button>
        </Box>

        <DesignTable
          designs={data}
          onDelete={(id: string) => handleDelete(id)}
          onEdit={(design: Design) => handleEdit(design)}
        />

        <DesignFormModal isOpen={isOpen} onClose={handleClose} onSave={(data: DesignForm) => handleSave(data)} />
      </Box>
    </DesignContext.Provider>
  )
}

export default DesignList;
