import { Box, Grid, GridItem, Heading, Text } from '@chakra-ui/react';
import { isRouteErrorResponse, useRouteError } from 'react-router-dom'
import Header from '../layout/Header';
import Sidebar from '../layout/Sidebar/Sidebar';

const ErrorPage = () => {
    const error = useRouteError();
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