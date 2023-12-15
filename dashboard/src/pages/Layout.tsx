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

import Sidebar from '../layout/Sidebar/Sidebar'
import { Outlet } from 'react-router-dom';
import Header from '../layout/Header';
import { Box, Grid, GridItem, useDisclosure } from '@chakra-ui/react';
import bannerBg from '../assets/bg-image.avif';
// import bannerBg from '../assets/bg-image-1.webp';

const MainLayout = () => {
  const { isOpen, onOpen, onClose } = useDisclosure();

    return (
      <>
        <Box
          bgImage={bannerBg}
          position="absolute"
          minH='40vh'
          w='100%'
          bgPos="center"
          bgSize="cover"
        ></Box>

        <Box
          display="flex"
          flexDirection="column"
          bgColor="#ebebeb"
          height="100%"
        >
          <Box height="50px">
              <Header onOpen={onOpen}/>
          </Box>

          <Box flex="1" height="calc(100% - 50px)">
            <Box>
                <Sidebar onClose={onClose} isOpen={isOpen}/>
            </Box>

            <Box padding='20px' overflowY="auto" height="100%" className="outlet">
              <Outlet />
            </Box>
          </Box>
        </Box>
      </>
    );
};

export default MainLayout;