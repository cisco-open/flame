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

import { MenuItem } from "./types";
import DashboardOutlinedIcon from '@mui/icons-material/DashboardOutlined';
import WorkOutlineOutlinedIcon from '@mui/icons-material/WorkOutlineOutlined';
import DesignServicesOutlinedIcon from '@mui/icons-material/DesignServicesOutlined';
import FolderOpenOutlinedIcon from '@mui/icons-material/FolderOpenOutlined';

const menuItems: MenuItem[] = [
  {
    id: 'dashboard',
    title: 'DASHBOARD',
    type: 'item',
    url: '/',
    icon: <DashboardOutlinedIcon sx={{ color: 'gray' }} fontSize="inherit"/>
  },
  {
    id: 'design',
    title: 'DESIGNS',
    type: 'item',
    url: '/design',
    icon: <DesignServicesOutlinedIcon sx={{ color: 'gray' }} fontSize="inherit"/>
  },
  {
    id: 'jobs',
    title: 'JOBS',
    type: 'item',
    url: '/jobs',
    icon: <WorkOutlineOutlinedIcon sx={{ color: 'gray' }} fontSize="inherit"/>
  },
  {
    id: 'datasets',
    title: 'DATASETS',
    type: 'item',
    url: '/datasets',
    icon: <FolderOpenOutlinedIcon sx={{ color: 'gray' }} fontSize="inherit"/>
  },
];

export default menuItems;