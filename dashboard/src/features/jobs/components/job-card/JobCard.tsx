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

import { Box, useColorModeValue, Image, Text, Stack, Heading } from '@chakra-ui/react'

const IMAGE =
  'https://images.unsplash.com/photo-1518051870910-a46e30d9db16?ixlib=rb-1.2.1&ixid=MXwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHw%3D&auto=format&fit=crop&w=1350&q=80'

const JobCard = () => {
  return (
    <Box
    role={'group'}
    p={6}
    maxW={'330px'}
    w={'full'}
    bg={useColorModeValue('white', 'gray.800')}
    boxShadow={'2xl'}
    rounded={'lg'}
    pos={'relative'}
    zIndex={1}>
    <Box
      rounded={'lg'}
      mt={-12}
      pos={'relative'}
      height={'230px'}
      _after={{
        transition: 'all .3s ease',
        content: '""',
        w: 'full',
        h: 'full',
        pos: 'absolute',
        top: 5,
        left: 0,
        backgroundImage: `url(${IMAGE})`,
        filter: 'blur(15px)',
        zIndex: -1,
      }}
      _groupHover={{
        _after: {
          filter: 'blur(20px)',
        },
      }}>
      <Image
        rounded={'lg'}
        height={230}
        width={282}
        objectFit={'cover'}
        src={IMAGE}
        alt="#"
      />
    </Box>
    <Stack pt={10} align={'center'}>
      <Text color={'gray.500'} fontSize={'sm'} textTransform={'uppercase'}>
        End Time
      </Text>
      <Heading fontSize={'2xl'} fontFamily={'body'} fontWeight={500}>
        Nice Chair, pink
      </Heading>
      <Stack direction={'row'} align={'center'}>
        <Text fontWeight={800} fontSize={'xl'}>
          $57
        </Text>
        <Text textDecoration={'line-through'} color={'gray.600'}>
          $199
        </Text>
      </Stack>
    </Stack>
  </Box>
  )
}

export default JobCard