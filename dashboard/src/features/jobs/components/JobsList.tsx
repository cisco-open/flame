import { TableContainer, Table, Thead, Tr, Th, Tbody, Td, Box, Tooltip, useDisclosure } from "@chakra-ui/react";
import { Job } from "../../../entities/Job";
import PlayCircleOutlineIcon from '@mui/icons-material/PlayCircleOutline';
import StopCircleIcon from '@mui/icons-material/StopCircle';
import useJobs from "../hooks/useJobs";
import { useNavigate } from "react-router-dom";
import React, { useState } from "react";
import EditOutlinedIcon from '@mui/icons-material/EditOutlined';
import DeleteOutlineOutlinedIcon from '@mui/icons-material/DeleteOutlineOutlined';
import ConfirmationDialog from "../../../components/confirmation-dialog/ConfirmationDialog";

const columns = ['Name', 'State', ''];

interface Props {
  openJobModal: (job: Job) => void;
}

const JobsList = ({ openJobModal }: Props) => {
  const [jobId, setJobId] = useState('');
  const navigate = useNavigate();
  const { data: jobs, updateStatusMutation, deleteMutation } = useJobs(jobId);
  const { isOpen, onOpen, onClose } = useDisclosure();

  const onStartClick = (event: React.MouseEvent, job: Job) => {
    event.stopPropagation();
    setJobId(job.id);

    updateStatusMutation.mutate({
      id: job.id,
      state: 'starting'
    })
  }

  const onStopClick = (event: React.MouseEvent, job: Job) => {
    event.stopPropagation();
    setJobId(job.id);

    updateStatusMutation.mutate({
      id: job.id,
      state: 'stopping'
    })
  }

  const openConfirmationModal = (event: React.MouseEvent, jobId: string) => {
    event.stopPropagation();
    setJobId(jobId);

    onOpen();
  }

  const onDelete = () => {
    deleteMutation.mutate();
    onClose();
  }

  const handleConfirmationClose = () => {
    setJobId('');
    onClose();
  }

  const goToJobDetails = (job: Job) => {
    navigate(`/jobs/${job.id}`)
  }

  const onEditClick = (event: React.MouseEvent, job: Job) => {
    event.stopPropagation();

    openJobModal(job);
  }

  return (
    // <SimpleGrid columns={3} spacing="20px">
    //   {jobs?.map(sjob => 
    //     <JobCard />
    //   )}
    // </SimpleGrid>
    <TableContainer flex={1} overflowY="auto">
        <Table variant='simple' fontSize={12} size="sm">
        <Thead>
            <Tr>
                {columns.map(column => <Th key={column}>{column}</Th>)}
            </Tr>
        </Thead>

        <Tbody>
            {jobs?.map((job: Job) =>
            <Tr height="50px" key={job.id} cursor="pointer" onClick={() => goToJobDetails(job)}>
                <Td>{job.name}</Td>

                <Td>{job.state}</Td>

                <Td>
                <Box display="flex" gap="10px" justifyContent="flex-end">
                    <Tooltip label="Edit" fontSize="inherit">
                      <EditOutlinedIcon onClick={(event) => onEditClick(event, job)} cursor="pointer" fontSize="small"/>
                    </Tooltip>

                    <Tooltip label={job.state !== 'running' ? 'Start Job' : 'Stop Job'} fontSize="inherit">
                      {
                        job.state !== 'running' ?
                          <PlayCircleOutlineIcon onClick={(event) => onStartClick(event, job)} cursor="pointer" fontSize="small"/> :
                          <StopCircleIcon onClick={(event) => onStopClick(event, job)} cursor="pointer" fontSize="small"/>
                      }
                    </Tooltip>

                    <Tooltip label="Delete">
                        <DeleteOutlineOutlinedIcon onClick={(event) => openConfirmationModal(event, job.id)} cursor="pointer" fontSize="small"/>
                    </Tooltip>
                </Box>
                </Td>
            </Tr>
            )}
        </Tbody>
        </Table>

        <ConfirmationDialog
          actionButtonLabel={'Delete'}
          message={'Are sure you want to delete this job?'}
          buttonColorScheme={'red'}
          isOpen={isOpen}
          onClose={handleConfirmationClose}
          onAction={onDelete}
        />
    </TableContainer>
  )
}

export default JobsList;