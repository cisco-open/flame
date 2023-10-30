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