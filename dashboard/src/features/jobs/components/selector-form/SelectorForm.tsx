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

import { Box, SimpleGrid, FormControl, FormLabel, Text, Select, Input } from '@chakra-ui/react'
import { useContext, useEffect, useState } from 'react';
import { ArgsGroup } from '../../../../entities/JobForm';
import { selector, SELECTOR_DEFAULT_OPTIONS, SELECTOR_OPTION, SELECTOR_OPTIONS } from '../../constants';
import { JobContext } from '../../JobContext';
import { createEntityKwargsPayload, getSelectorsFromJob } from '../../utils';

const DEFAULT_SELECTOR_GROUPS = [{ arg: '', value: '', id: 1 }];

const SelectorForm = () => {
  const [ selectedSelector, setSelectedSelector ] = useState<string>(selector.DEFAULT);
  const [ selectorKwargs, setSelectorKwargs ] = useState<any[]>();
  const [ selectorGroups, setSelectorGroups ] = useState<any[]>(DEFAULT_SELECTOR_GROUPS);
  const { setSelectorKwargsPayload } = useContext(JobContext);
  const { job } = useContext(JobContext);

  const selectorOptions = SELECTOR_OPTIONS;

  useEffect(() => {
    setSelectorKwargs([...SELECTOR_DEFAULT_OPTIONS[selectedSelector as keyof typeof SELECTOR_OPTION]])
    setSelectorGroups([...SELECTOR_DEFAULT_OPTIONS[selectedSelector as keyof typeof SELECTOR_OPTION]]);
  }, [selectedSelector]);

  useEffect(() => {
    if (!job) { return; }
    setSelectedSelector(getSelectorsFromJob(job));
  }, [job])

  useEffect(() => {
    const payload = createEntityKwargsPayload(selectedSelector, selectorGroups);
    setSelectorKwargsPayload(payload);
  }, [selectorGroups, setSelectorKwargsPayload, selectedSelector]);



  const onSelectorChange = (event: any) => {
    setSelectedSelector(SELECTOR_OPTIONS.find(option => option.value === event.target.value)?.value || '');
  }

  const setSelectorValue = (event: any, index: number, paramEnum: ArgsGroup) => {
    const targetParameter = selectorGroups[index];

    paramEnum === ArgsGroup.key ?
      targetParameter.arg = event.target.value :
      targetParameter.value = event.target.value;
      selectorGroups[index] = targetParameter;
    setSelectorGroups([...selectorGroups]);
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
      <Text as="h4" textAlign="center">Selector</Text>

      <SimpleGrid columns={2} spacing="20px">
        <FormControl display="flex" justifyContent="space-between">
          <FormLabel fontSize="12px" flex="1">Sort</FormLabel>

          <Select
            backgroundColor="white"
            size="xs"
            onChange={(event) => onSelectorChange(event)}
            value={selectedSelector}
          >
            {
              selectorOptions.map(param => <option key={param.id} value={param.value}>{param.label}</option>)
            }
          </Select>
        </FormControl>

        {(selectorKwargs?.length || []) > 0 &&
          <Box display="flex" flexDirection="column" gap="20px" justifyContent="center" alignItems="center">
            <Box width="100%" display="flex" flexDirection="column" gap="10px">
              <Box display="flex" justifyContent="space-around">
                <Text as="label" fontSize="12px">Key</Text>
                <Text as="label" fontSize="12px">Value</Text>
              </Box>
              {
                selectorGroups.map((group, index) =>
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
                      onChange={(event) => setSelectorValue(event, index, ArgsGroup.value)}
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

export default SelectorForm