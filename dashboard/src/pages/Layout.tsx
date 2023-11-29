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
import { Grid, GridItem } from '@chakra-ui/react';

const MainLayout = () => {
    return (
      <Grid
        templateAreas={`
            'nav header'
            'nav main'
        `}
        gridTemplateRows={'50px 1fr'}
        gridTemplateColumns={'150px 1fr'}
        height='100vh'
      >
        <GridItem area="header">
            <Header />
        </GridItem>

        <GridItem area="nav">
            <Sidebar />
        </GridItem>

        <GridItem area="main" padding='20px' height="100%" overflowY="auto">
          <Outlet />
        </GridItem>
      </Grid>
    );
};

export default MainLayout;