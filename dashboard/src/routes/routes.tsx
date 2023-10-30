import { createBrowserRouter } from "react-router-dom";
import DashboardPage from "../pages/DashboardPage";
import ErrorPage from "../pages/ErrorPage";
import MainLayout from "../pages/Layout";
import DesignPage from "../features/design/DesignPage";
import DesignDetailsPage from "../features/design-details/DesignDetailsPage";
import JobPage from "../features/jobs/JobPage";
import JobDetailsPage from "../features/job-details/JobDetailsPage";
import DatasetsPage from "../features/datasets/DatasetsPage";

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
            }
        ]
    }
]);

export default router;