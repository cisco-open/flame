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

import { Spinner, Flex, Text } from '@chakra-ui/react';

interface Props {
  size?: "xl" | "sm" | "md" | "lg" | "xs";
  message?: string;
}

const Loading = ({ size = 'lg', message = 'Loading' }: Props) => {
  return (
    <Flex direction="column" align="center" justify="center" height="100%" gap="20px">
      <Spinner size={size} colorScheme="primary" />

      <Text as="p" fontSize="12px">{message}</Text>
    </Flex>
  );
};

export default Loading;