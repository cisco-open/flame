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

import { Modal, ModalOverlay, ModalContent, ModalHeader, ModalCloseButton, ModalBody, FormControl, FormLabel, Input, Button, ModalFooter } from '@chakra-ui/react'
import { useRef } from 'react';
import * as yup from 'yup';
import { yupResolver } from "@hookform/resolvers/yup";
import { useForm } from "react-hook-form";
import { ComputeFormData } from '../types';

interface Props {
  isOpen: boolean;
  onClose: () => void;
  onSave: (data: ComputeFormData) => void;
}

const ComputeFormModal = ({ isOpen, onClose, onSave }: Props) => {
  const initialRef: React.MutableRefObject<null> = useRef(null);

  const schema = yup.object().shape({
    adminId: yup.string().required(),
    region: yup.string().required(),
    apiKey: yup.string().required(),
    computeId: yup.string().required(),
  });

  const { register, handleSubmit, formState, reset } = useForm({
    resolver: yupResolver(schema),
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
      <ModalContent>
        <ModalHeader textAlign="center">CREATE COMPUTE</ModalHeader>

        <ModalCloseButton />

        <ModalBody display="flex" flexDirection="column" gap="20px">
          <FormControl mt={4}>
            <FormLabel fontSize="12px">Compute ID</FormLabel>
            <Input size="xs" placeholder='Compute Id' {...register('computeId')}/>
          </FormControl>

          <FormControl>
            <FormLabel fontSize="12px">Admin ID</FormLabel>
            <Input size="xs" placeholder='Admin ID' {...register('adminId')} />
          </FormControl>

          <FormControl>
            <FormLabel fontSize="12px">Region</FormLabel>
            <Input size="xs" placeholder='Region' {...register('region')}/>
          </FormControl>

          <FormControl>
            <FormLabel fontSize="12px">API Key</FormLabel>
            <Input size="xs" placeholder='API Key' {...register('apiKey')}/>
          </FormControl>
        </ModalBody>

        <ModalFooter>
          <Button onClick={handleSubmit(onSave)} colorScheme='primary' size="xs" mr={3} isDisabled={!formState.isValid}>
            Save
          </Button>
          <Button onClick={handleClose} colorScheme="secondary" variant="outline" size="xs">Cancel</Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  )
}

export default ComputeFormModal;