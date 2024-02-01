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

import { createBrowserRouter } from "react-router-dom";
import DashboardPage from "../pages/DashboardPage";
import ErrorPage from "../pages/ErrorPage";
import MainLayout from "../pages/Layout";
import DesignPage from "../features/design/DesignPage";
import DesignDetailsPage from "../features/design-details/DesignDetailsPage";
import JobPage from "../features/jobs/JobPage";
import JobDetailsPage from "../features/job-details/JobDetailsPage";
import DatasetsPage from "../features/datasets/DatasetsPage";
import ComputesPage from "../features/computes/ComputesPage";

const router = createBrowserRouter([
    {
        path: '/',
        element: <MainLayout />,
        errorElement: <ErrorPage />,
        children: [
            {
                index: true,
                element: <DashboardPage />
            },
            {
                path: '/jobs',
                element: <JobPage />
            },
            {
                path: '/jobs/:id',
                element: <JobDetailsPage />
            },
            {
                path: '/design',
                element: <DesignPage />,
            },
            {
                path: '/design/:id',
                element: <DesignDetailsPage />
            },
            {
                path: '/datasets',
                element: <DatasetsPage />
            },
            {
                path: '/computes',
                element: <ComputesPage />
            }
        ]
    }
]);

export default router;