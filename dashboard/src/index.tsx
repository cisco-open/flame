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

export const colors = {
  primary: {
    lighter: '#e6efdc',
    light: '#b4d095',
    normal: '#283618',
    darker: '#1a2310',
  },
  secondary: {
    lighter: '#e8e8e3',
    light: '#bbbbaa',
    normal: '#B7B7A4',
    darker: '#1c1c17',
  },
  grayGreen: {
    lighter: '#e6e6e6',
    light: '#b3b3b3',
    normal: '#D4D4D4',
    darker: '#b3b3b3',
  },
  offWhite: {
    lighter: '#f9f9f7',
    light: '#f1f0ed',
    normal: '#F0EFEB',
    darker: '#d6d3c8',
  }
}

// Hex codes: dark green-gray #283618, light gray-green #B7B7A4, soft gray #D4D4D4, off-white #F0EFEB

// export const colors = {
//   secondary: {
//     lighter: '#babac5',
//     light: '#8b8d9e',
//     normal: '#AAABB8',
//     darker: '#616274',
//   },
//   primary: {
//     lighter: '#ababd3',
//     light: '#7474b6',
//     normal: '#2C2C54',
//     darker: '#0f0f1c',
//   },
//   skyBlue: {
//     lighter: '#ebf6f9',
//     light: '#c4e3ed',
//     normal: '#A9D6E5',
//     darker: '#76bed6',
//   },
//   lightGray: {
//     lighter: '#eef6f4',
//     light: '#cce5df',
//     normal: '#E2E2E2',
//     darker: '#88c3b4',
//   }
// }


const theme = extendTheme({
  components: {
    MultiSelect: MultiSelectTheme
  },
  fonts: {
    body: 'Montserrat'
  },
  colors: {
    primary: {
      50: colors.primary.lighter,
      100: colors.primary.light,
      // ... define other shades of secondary color
      500: colors.primary.normal, // Replace this with your desired secondary color
      // ... define other shades of secondary color
      900: colors.primary.darker,
    },
    secondary: {
      50: colors.secondary.lighter,
      100: colors.secondary.light,
      // ... define other shades of primary color
      500: colors.secondary.normal, // Replace this with your desired primary color
      // ... define other shades of primary color
      900: colors.secondary.darker,
    },
  },
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
