import { Box, Button, useDisclosure, Text } from '@chakra-ui/react';
import AddIcon from '@mui/icons-material/Add';
import { useEffect, useState } from 'react';
import { Job } from '../../entities/Job';
import JobFormModal from './components/job-form-modal/JobFormModal';
import JobsList from './components/JobsList';
import useJob from './hooks/useJob';

const JobPage = () => {
  const { isOpen, onOpen, onClose } = useDisclosure();
  const [ jobInEdit, setJobInEdit ] = useState<Job | null>(null);
  const { data: job } = useJob(jobInEdit?.id || '');

  useEffect(() => {
    if (job) {
      onOpen();
    }
  }, [job]);

  const handleClose = () => {
    setJobInEdit(null);
    onClose();
  };

  const handleEditJob = (job: Job) => {
    setJobInEdit(job);
  }

  return (
    <Box gap={5} display="flex" flexDirection="column" height="100%" overflow="hidden">
      <Box display="flex" alignItems="center" justifyContent="space-between">
        <Text as="h1" fontWeight="bold">JOBS</Text>

        <Button leftIcon={<AddIcon fontSize="small" />} onClick={onOpen} alignSelf="flex-end" variant='outline' size="xs" colorScheme="teal">Create New</Button>
      </Box>

      <JobsList openJobModal={(job: Job) => handleEditJob(job)} />

      { isOpen && <JobFormModal isOpen={isOpen} job={job} onClose={handleClose}/> }
    </Box>
  )
}

export default JobPage;