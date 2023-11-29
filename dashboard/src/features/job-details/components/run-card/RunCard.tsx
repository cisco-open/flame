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

import { Center, Box, useColorModeValue, Stack, List, ListItem, Text, ListIcon, Button } from '@chakra-ui/react'
import { useEffect, useState } from 'react';
import { Run } from '../../../../entities/JobDetails';
import { format } from 'date-fns'

interface Props {
  run: Run;
}

const RunCard = ({ run }: Props) => {
  const [ mappedRun, setMappedRun ] = useState<Run>();

  useEffect(() => {
    setMappedRun({
      ...run,
      startDate: format(run.info.start_time, 'MM/dd/yyyy hh:mm a'),
      endDate: format(run.info.end_time, 'MM/dd/yyyy hh:mm a')
    })
  }, [run])
  return (
      <Box
        maxW={'330px'}
        w={'full'}
        bg={useColorModeValue('white', 'gray.800')}
        boxShadow={'lg'}
        rounded={'md'}
        overflow={'hidden'}>
        <Stack
          textAlign={'center'}
          p={6}
          color={useColorModeValue('gray.800', 'white')}
          align={'center'}>
          <Text
            fontSize={'xs'}
            fontWeight={700}
            bg={useColorModeValue('green.50', 'green.900')}
            p={2}
            px={3}
            color={'green.500'}
            rounded={'full'}>
            {mappedRun?.info.run_name}
          </Text>
          <Stack direction={'row'} align={'center'} justify={'center'}>
            <Text fontWeight={800}>
              {mappedRun?.info.status}
            </Text>
          </Stack>
        </Stack>

        <Box px={6} py={10}>
          <List spacing={3}>
            <ListItem fontSize="xs" display="flex" gap="5px">
              <Text fontWeight="bold">Start:</Text>{mappedRun?.startDate}
            </ListItem>
            <ListItem fontSize="xs" display="flex" gap="5px">
              <Text fontWeight="bold">End:</Text> {mappedRun?.endDate}
            </ListItem>
            <ListItem fontSize="xs" display="flex" gap="5px">
              <Text fontWeight="bold">Lifecycle Stage:</Text> {run.info.lifecycle_stage}
            </ListItem>
          </List>

          <Button
            mt={10}
            w={'full'}
            bg={'blue.400'}
            color={'white'}
            rounded={'xl'}
            boxShadow={'0 5px 20px 0px rgb(72 88 243 / 43%)'}
            _hover={{
              bg: 'blue.500',
            }}
            _focus={{
              bg: 'blue.500',
            }}>
            View Details
          </Button>
        </Box>
      </Box>
  )
}

export default RunCard