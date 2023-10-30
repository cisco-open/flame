import { Box, SimpleGrid, Text } from '@chakra-ui/react';
import { Parameter } from '../../../../entities/JobDetails';
import './RunParameter.css';

interface Props {
  parameters: Parameter[] | undefined;
}

const RunParameters = ({ parameters }: Props) => {
  if (!parameters) {
    return <Box className="parameters-zero-state">
      <Text>No data reported</Text>
    </Box>
  }

  return (
    <SimpleGrid columns={4} spacing="20px">{
      parameters?.map(param => <Box className="parameter" key={param.key}>
        <Text>{param.key}</Text>
        <Text>{param.value}</Text>
      </Box>)  
    }</SimpleGrid>
  )
}

export default RunParameters