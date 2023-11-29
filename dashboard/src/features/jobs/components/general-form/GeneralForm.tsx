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

import { Box, FormControl, FormLabel, Input, Select, SimpleGrid } from '@chakra-ui/react'
import React, { useContext, useEffect } from 'react'
import { Design } from '../../../../entities/Design';
import {UseFormRegister } from 'react-hook-form';
import { FormFields } from '../job-form-modal/JobFormModal';
import { JobContext } from '../../JobContext';

interface Props {
  designs: Design[] | undefined;
  setSelectedDesignId: (id: string) => void;
  register: UseFormRegister<FormFields>;
  setValue: (name: string, value: any) => void;
  backendOptions: { name: string, id: number }[];
  selectedDesignId: string;
}

const GeneralForm = ({ designs, backendOptions, selectedDesignId, setSelectedDesignId, register, setValue }: Props) => {
  const { job } = useContext(JobContext);

  useEffect(() => {
    if (job) {
      setSelectedDesignId(job.designId);
      setValue('name', job.name);
      setValue('backend', job.backend);
      setValue('maxRunTime', job.maxRunTime);
    }
  }, [job]);

  const onDesignSelect = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const designId = designs?.find(design => design.id === event.target.value)?.id;
    setSelectedDesignId(designId ?? '');
  };
  return (
    <Box display="flex" flexDirection="column" gap="20px" height="100%" overflow="hidden" overflowY="auto" alignItems="center" padding="10px">
      <SimpleGrid
        columns={1}
        spacing="20px"
        borderRadius="4px"
        padding="10px"
        width="50%"
      >
        <FormControl>
          <FormLabel fontSize="12px">Name</FormLabel>

          <Input
            size="xs"
            placeholder='Name'
            {...register('name')}
          />
        </FormControl>

        <FormControl>
          <FormLabel fontSize="12px">Design</FormLabel>

          <Select
            size="xs"
            placeholder='Select option'
            value={selectedDesignId}
            onChange={(event) => onDesignSelect(event)}
          >
            {designs?.map(design =>
              <option key={design.id} value={design.id}>{design.name}</option>
            )}
          </Select>
        </FormControl>

        <FormControl>
          <FormLabel fontSize="12px">Backend</FormLabel>

          <Select
            size="xs"
            placeholder='Select option'
            {...register('backend')}
          >
            {backendOptions?.map(option =>
              <option key={option.id} value={option.name}>{option.name}</option>
            )}
          </Select>
        </FormControl>

        <FormControl>
          <FormLabel fontSize="12px">Timeout (in seconds)</FormLabel>

          <Input
            size="xs"
            placeholder='Select option'
            {...register('maxRunTime')}
          />
        </FormControl>
      </SimpleGrid>
    </Box>
  )
}

export default GeneralForm