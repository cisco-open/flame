import React from 'react';
import ReactDOM from 'react-dom/client';
import reportWebVitals from './reportWebVitals';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import { RouterProvider } from 'react-router-dom';
import router from './routes/routes';
import { ChakraProvider, extendTheme } from '@chakra-ui/react';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import {  MultiSelectTheme } from 'chakra-multiselect';
import './index.css';
import { ReactFlowProvider } from 'reactflow';

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false
    }
  }
});

const MuiTheme = createTheme({
  palette: {
    mode: "light",
  },
});


const theme = extendTheme({
  components: {
    MultiSelect: MultiSelectTheme
  },
  fonts: {
    body: 'Montserrat'
  }
})


root.render(
  <React.StrictMode>
    <ReactFlowProvider>
      <ThemeProvider theme={MuiTheme}>
        <ChakraProvider theme={theme}>
            <QueryClientProvider client={queryClient}>
              <RouterProvider router={router} />

              <ReactQueryDevtools />
            </QueryClientProvider>
        </ChakraProvider>
      </ThemeProvider>
    </ReactFlowProvider>
  </React.StrictMode>
);

reportWebVitals();
