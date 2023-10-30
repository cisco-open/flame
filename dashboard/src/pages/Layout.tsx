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