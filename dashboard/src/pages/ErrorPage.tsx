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

import { Box, Grid, GridItem, Heading, Text, useDisclosure } from '@chakra-ui/react';
import { isRouteErrorResponse, useRouteError } from 'react-router-dom'
import Header from '../layout/Header';
import Sidebar from '../layout/Sidebar/Sidebar';

const ErrorPage = () => {
    const error = useRouteError();
    const { isOpen, onOpen, onClose } = useDisclosure();
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
            <Header onOpen={onOpen}/>
        </GridItem>

        <GridItem area="nav">
            <Sidebar onClose={onClose} isOpen={isOpen}/>
        </GridItem>

        <GridItem area="main" paddingX='5px'>
        <Box>
            <Heading>Oops</Heading>
            <Text>{ isRouteErrorResponse(error) ? 'This page does not exist' : 'An unexpected error occured.' }</Text>
        </Box>
        </GridItem>
    </Grid>
  )
}

export default ErrorPage