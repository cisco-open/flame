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

import { Box, SimpleGrid, FormControl, FormLabel, Select, Text, Input, Button } from '@chakra-ui/react'
import { useContext, useEffect, useState } from 'react'
import { ArgsGroup } from '../../../../entities/JobForm';
import { optimizer, OPTIMIZER_DEFAULT_OPTIONS, OPTIMIZER_OPTION, OPTIMIZER_OPTIONS } from '../../constants';
import { createEntityKwargsPayload } from '../../utils';
import { JobContext } from '../../JobContext';

export interface KwargPayload {
  sort: 'string',
  kwargs: {
    [key: string]: string | number,
  }
}

export interface OptimizerGroup {
  arg: string,
  value: string,
  id: number,
}

const DEFAULT_OPTIMIZER_GROUPS = [{ arg: '', value: '', id: 1 }];

const OptimizerForm = () => {
  const [ selectedOptimizer, setSelectedOptimizer ] = useState<string>(optimizer.FEDAVG);
  const [ optimizerKwargs, setOptimizerKwargs ] = useState<any[]>();
  const [ optimizerGroups, setOptimizerGroups ] = useState<any[]>(DEFAULT_OPTIMIZER_GROUPS);
  const { setOptimizerKwargsPayload, job } = useContext(JobContext);

  const optimizerOptions = OPTIMIZER_OPTIONS;

  useEffect(() => {
    setOptimizerKwargs([...OPTIMIZER_DEFAULT_OPTIONS[selectedOptimizer as keyof typeof OPTIMIZER_OPTION]]);
    setOptimizerGroups([...OPTIMIZER_DEFAULT_OPTIONS[selectedOptimizer as keyof typeof OPTIMIZER_OPTION]]);
  }, [selectedOptimizer]);

  useEffect(() => {
    const payload = createEntityKwargsPayload(selectedOptimizer, optimizerGroups);
    setOptimizerKwargsPayload(payload);
  }, [optimizerGroups]);

  useEffect(() => {
    // @TODO - adjust mappings for complex selectors
    if (!job) { return; }
    setSelectedOptimizer(job.modelSpec.optimizer.sort);
  }, [job])

  const onOptimizerChange = (event: any) => {
    setSelectedOptimizer(OPTIMIZER_OPTIONS.find(option => option.value === event.target.value)?.value || '');
  }

  const setOptimizerValue = (event: any, index: number, paramEnum: ArgsGroup) => {
    const targetParameter = optimizerGroups[index];

    paramEnum === ArgsGroup.key ?
      targetParameter.arg = event.target.value :
      targetParameter.value = event.target.value;
      optimizerGroups[index] = targetParameter;
    setOptimizerGroups([...optimizerGroups]);
  }

  return (
    <Box
      backgroundColor="#f2f2f2"
      borderRadius="4px"
      padding="10px"
      display="flex"
      flexDirection="column"
      gap="20px"
      width="70%"
    >
      <Text as="h4" textAlign="center">Optimizer</Text>

      <SimpleGrid columns={2} spacing="20px">
        <FormControl display="flex" justifyContent="space-between">
          <FormLabel fontSize="12px" flex="1">Sort</FormLabel>

          <Select
            backgroundColor="white"
            size="xs"
            placeholder="key"
            onChange={(event) => onOptimizerChange(event)}
            value={selectedOptimizer}
          >
            {
              optimizerOptions.map(param => <option key={param.id} value={param.value}>{param.label}</option>)
            }
          </Select>
        </FormControl>

        {
          (optimizerKwargs?.length || []) > 0 &&
          <Box display="flex" flexDirection="column" gap="20px" justifyContent="center" alignItems="center">
            <Box width="100%" display="flex" flexDirection="column" gap="10px">
              <Box display="flex" justifyContent="space-around">
                <Text as="label" fontSize="12px">Key</Text>
                <Text as="label" fontSize="12px">Value</Text>
              </Box>
              {
                optimizerGroups.map((group, index) =>
                  <FormControl key={group.arg} display="flex" justifyContent="space-between" gap="10px" alignItems="center">

                    <Input
                      backgroundColor="white"
                      size="xs"
                      value={group.arg}
                      readOnly
                      placeholder="value"
                    />

                    <Input
                      backgroundColor="white"
                      size="xs"
                      value={group.value}
                      placeholder="value"
                      onChange={(event) => setOptimizerValue(event, index, ArgsGroup.value)}
                    />
                  </FormControl>
                )
              }
            </Box>
          </Box>
        }
      </SimpleGrid>
    </Box>
  )
}

export default OptimizerForm