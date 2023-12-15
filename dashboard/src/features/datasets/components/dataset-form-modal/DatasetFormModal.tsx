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

import { useForm } from 'react-hook-form';
import * as yup from 'yup';
import { yupResolver } from '@hookform/resolvers/yup';
import { Button, Checkbox, FormControl, FormLabel, Input, Modal, ModalBody, ModalCloseButton, ModalContent, ModalFooter, ModalHeader, ModalOverlay, SimpleGrid, Textarea } from '@chakra-ui/react';
import { useEffect, useRef } from 'react';
import './DatasetFormModal.css';

interface Props {
  isOpen: boolean;
  onClose: () => void;
  onSave: (data: any) => void;
  isSaveSuccess: boolean;
}

const DatasetForm = ({ isOpen, isSaveSuccess, onClose, onSave }: Props) => {
  const initialRef: React.MutableRefObject<null> = useRef(null);

  useEffect(() => {
    if (isSaveSuccess) {
      reset();
    }
  }, [isSaveSuccess])

  const schema = yup.object().shape({
    name: yup.string().required(),
    url: yup.string().required(),
    description: yup.string(),
    computeId: yup.string(),
    isPublic: yup.boolean(),
  });

  const { register, handleSubmit, formState: { isValid }, reset } = useForm({
    resolver: yupResolver(schema)
  });

  const handleClose = () => {
    onClose();
    reset();
  }

  return (
    <Modal
      initialFocusRef={initialRef}
      isOpen={isOpen}
      onClose={handleClose}
    >
      <ModalOverlay />
      <ModalContent className="dataset-form">
        <ModalHeader textAlign="center">Create Dataset</ModalHeader>

        <ModalCloseButton />

        <ModalBody pb={6} display="flex" flexDirection="column" gap="10px">
          <SimpleGrid
            columns={2}
            spacing="20px"
          >
            <FormControl>
              <FormLabel fontSize="12px">Name</FormLabel>
              <Input size="xs" placeholder='Name' {...register('name')}/>
            </FormControl>

            <FormControl>
              <FormLabel fontSize="12px">URL</FormLabel>
              <Input size="xs" placeholder='URL' {...register('url')} />
            </FormControl>

            <FormControl>
              <FormLabel fontSize="12px">Compute ID</FormLabel>
              <Input size="xs" placeholder='Compute ID' {...register('computeId')} />
            </FormControl>

            <FormControl>
              <FormLabel fontSize="12px" visibility="hidden">Is Public</FormLabel>
              <Checkbox {...register('isPublic')} colorScheme="primary" size="xs">Is Public</Checkbox>
            </FormControl>

            <FormControl>
              <FormLabel fontSize="12px">Description</FormLabel>
              <Textarea size="xs" placeholder='Description' {...register('description')} />
            </FormControl>
          </SimpleGrid>
        </ModalBody>

        <ModalFooter>
          <Button onClick={handleSubmit(onSave)} colorScheme='primary' size="xs" mr={3} isDisabled={!isValid}>
            Save
          </Button>
          <Button onClick={handleClose} colorScheme="secondary" size="xs" variant="outline" >Cancel</Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  )
}

export default DatasetForm