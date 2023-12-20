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

import { Box, SimpleGrid, FormControl, FormLabel, Input, Button, Text, Select } from '@chakra-ui/react';
import AddIcon from '@mui/icons-material/Add';
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline';
import { useContext, useEffect } from 'react';
import { UseFormRegister } from 'react-hook-form';
import { ArgsGroup } from '../../../../entities/JobForm';
import { hyperparameters, HYPERPARAMETER_OPTIONS, HYPERPARAMETER_TYPE } from '../../constants';
import { JobContext } from '../../JobContext';
import { FormFields } from '../job-form-modal/JobFormModal';
import OptimizerForm from '../optimizer-form/OptimizerForm';
import SelectorForm from '../selector-form/SelectorForm';

interface Props {
  register: UseFormRegister<FormFields>;
  hyperParameters: { key: string, value: string, id: number, type?: HYPERPARAMETER_TYPE }[];
  setHyperParameters: (parameters: any) => void;
  setValue: (name: string, value: any) => void;
}

const ModelSpecForm = ({
  hyperParameters,
  register,
  setHyperParameters,
  setValue
}: Props) => {
  const { job } = useContext(JobContext);
  useEffect(() => {
    if (!job) { return; }
    setValue('baseModelName', job?.baseModel?.name);
    setValue('basemodelVersion', job?.baseModel?.version);
  }, [job]);

  const hyperparametersOptions = HYPERPARAMETER_OPTIONS;

  const setHyperparameterValue = (event: any, index: number, paramEnum: ArgsGroup) => {
    const targetParameter = hyperParameters[index];

    paramEnum === ArgsGroup.key ?
      targetParameter.key = event.target.value :
      targetParameter.value = event.target.value;

    if (event.target.value === hyperparameters.CUSTOM) {
      targetParameter.type = HYPERPARAMETER_TYPE.custom;
    }

    hyperParameters[index] = targetParameter;
    setHyperParameters(hyperParameters);
  }

  const addHyperParameter = () => {
    setHyperParameters([
      ...hyperParameters,
      { key: '', value: '', id: hyperParameters.length + 1 }
    ])
  }

  const removeHyperParameter = (id: number) => {
    setHyperParameters(hyperParameters.filter(parameter => parameter.id !== id));
  }

  return (
    <Box display="flex" flexDirection="column" gap="20px" height="100%" overflow="hidden" overflowY="auto" alignItems="center" padding="10px">
      <Box
        backgroundColor="#f2f2f2"
        borderRadius="4px"
        padding="10px"
        display="flex"
        flexDirection="column"
        gap="20px"
        width="70%"
      >
        <Text as="h4" textAlign="center">Base model</Text>

        <SimpleGrid
          columns={2}
          spacing="20px"
        >
          <FormControl display="flex" justifyContent="space-between">
            <FormLabel fontSize="12px" flex="1">Name</FormLabel>

            <Input
              backgroundColor="white"
              width="70%"
              size="xs"
              placeholder='name'
              {...register('basemodelName')}
            />
          </FormControl>

          <FormControl display="flex" justifyContent="space-between">
            <FormLabel fontSize="12px" flex="1">Version</FormLabel>

            <Input
              backgroundColor="white"
              width="70%"
              size="xs"
              placeholder='version'
              {...register('basemodelVersion')}
            />
          </FormControl>
        </SimpleGrid>
      </Box>

      <OptimizerForm  />

      <SelectorForm />

      <Box
        backgroundColor="#f2f2f2"
        borderRadius="4px"
        padding="10px"
        display="flex"
        flexDirection="column"
        gap="20px"
        alignItems="center"
        width="70%"
      >
        <Box>
          <Text as="h4" textAlign="center">Hyperparameters</Text>

          <Text fontSize="10px" as="label">Value can be single value or a list (comma separated values)</Text>
        </Box>

        <Box display="flex" justifyContent="space-around" width="100%">
          <Text as="label" fontSize="12px">Key</Text>
          <Text as="label" fontSize="12px">Value</Text>
        </Box>

        {hyperParameters.map((parameter, index) =>
          <FormControl key={parameter.id} display="flex" justifyContent="space-between" gap="20px">
            { parameter.type === HYPERPARAMETER_TYPE.custom ?
                <Input
                backgroundColor="white"
                size="xs"
                value={parameter.key}
                placeholder="value"
                onChange={(event) => setHyperparameterValue(event, index, ArgsGroup.key)}
              /> :
              <Select
                backgroundColor="white"
                size="xs"
                placeholder="key"
                onChange={(event) => setHyperparameterValue(event, index, ArgsGroup.key)}
                value={parameter.key}
              >
                {
                  hyperparametersOptions.map(param => <option key={param.id} value={param.value}>{param.label}</option>)
                }
              </Select>
            }

            <Input
              backgroundColor="white"
              size="xs"
              value={parameter.value}
              placeholder="value"
              onChange={(event) => setHyperparameterValue(event, index, ArgsGroup.value)}
            />

            <Button size="xs" className="delete-button" colorScheme="red" variant="outline" leftIcon={<DeleteOutlineIcon fontSize="small" />} onClick={() => removeHyperParameter(parameter.id)} />
          </FormControl>
        )}

        <Box>
           <Button
            size="xs"
            leftIcon={<AddIcon fontSize="small" />}
            onClick={addHyperParameter}
            colorScheme='primary'
            mr={3}
          >
            Add More
          </Button>
        </Box>
      </Box>
    </Box>
  )
}

export default ModelSpecForm