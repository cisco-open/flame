import { Box, Button } from "@chakra-ui/react";
import { useNavigate } from "react-router-dom";
import JobTopology from "./components/job-topology/JobTopology";
import ArrowBackIosIcon from '@mui/icons-material/ArrowBackIos';

export const fitViewOptions = {
  padding: 1,
  maxZoom: 4
}

const JobDetailsPage = () => {
  const navigate = useNavigate();

  return (
    <>
      <Button marginTop="2px" leftIcon={<ArrowBackIosIcon fontSize="small" />} onClick={() => navigate('/jobs')} variant='link' size="xs">Back</Button>
      <Box width="100%" height="100%">
        <JobTopology />
      </Box>
    </>
  )
}

export default JobDetailsPage;

