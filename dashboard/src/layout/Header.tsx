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

import { Image } from '@chakra-ui/react';
import { Box } from '@mui/material';
import avatarPlaceholder from '../assets/default-avatar.png';
import MenuOutlinedIcon from '@mui/icons-material/MenuOutlined';

interface Props {
  onOpen?: () => void;
}

const Header = ({ onOpen }: Props) => {
  return (
    <Box
      padding="5px 20px"
      borderBottom="1px solid rgba(58, 53, 65, 0.12)" 
      height="50px"
      display="flex"
      alignItems="center"
      justifyContent="space-between"
      zIndex="1"
      position="relative"
    >
      <MenuOutlinedIcon onClick={onOpen} fontSize="small" cursor="pointer"/>

      <Image
        src={avatarPlaceholder}
        height="30px"
        borderRadius="50%"
        cursor="pointer"
      />
    </Box>
  );
};

export default Header;
