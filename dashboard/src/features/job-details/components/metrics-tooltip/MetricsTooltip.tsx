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

import { Box } from '@chakra-ui/react';
import { time } from 'console';
import React, { useEffect } from 'react'

const MetricsTooltip = (props: any) => {
  if (props.active && props.payload && props.payload.length) {
    const { name, time, category } = props.payload[0].payload;

    return (
      <Box bgColor="white" padding="10px" border="1px solid gray" boxShadow="rgba(58, 53, 65, 0.42) 0px 4px 8px -4px">
        <p><strong>Category:</strong> {category}</p>
        <p><strong>Name:</strong> {name}</p>
        <p><strong>Time:</strong> {time} sec</p>
      </Box>
    );
  }

  return null;
}

export default MetricsTooltip