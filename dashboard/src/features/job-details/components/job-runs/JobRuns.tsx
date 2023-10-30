import { SimpleGrid } from '@chakra-ui/react';
import { useParams } from 'react-router-dom';
import { GetRunsPayload, RunViewType } from '../../../../entities/JobDetails';

interface Props {
  experimentId: string;
}



const JobRuns = ({ experimentId }: Props) => {
  const { id } = useParams();


  return (
    <>
      <SimpleGrid columns={4} spacing="20px">
        {
          // runsResponse?.runs.map(run =>
          //   <RunCard run={run} />
          // )
        }
      </SimpleGrid>
    </>
  )
}

export default JobRuns