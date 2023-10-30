import { Box, SimpleGrid, FormControl, FormLabel, Input, Button, Text, Select } from '@chakra-ui/react';
import AddIcon from '@mui/icons-material/Add';
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline';
import { useContext, useEffect } from 'react';
import { UseFormRegister } from 'react-hook-form';
import { ArgsGroup } from '../../../../entities/JobForm';
import { HYPERPARAMETER_OPTIONS } from '../../constants';
import { JobContext } from '../../JobContext';
import { FormFields } from '../job-form-modal/JobFormModal';
import OptimizerForm from '../optimizer-form/OptimizerForm';
import SelectorForm from '../selector-form/SelectorForm';

interface Props {
  register: UseFormRegister<FormFields>;
  hyperParameters: { key: string, value: string, id: number}[];
  setHyperParameters: (parameters: any) => void;
  setValue: (name: string, value: any) => void;
}

const ModelSpecForm = ({
  hyperParameters,
  register,
  setHyperParameters,
  setValue
}: Props) => {
  const { job } = useContext(JobContext);
  useEffect(() => {
    if (!job) { return; }
    setValue('baseModelName', job?.baseModel?.name);
    setValue('basemodelVersion', job?.baseModel?.version);
  }, [job]);

  const hyperparametersOptions = HYPERPARAMETER_OPTIONS;
  const setHyperparameterValue = (event: any, index: number, paramEnum: ArgsGroup) => {
    const targetParameter = hyperParameters[index];

    paramEnum === ArgsGroup.key ?
      targetParameter.key = event.target.value :
      targetParameter.value = event.target.value;
    hyperParameters[index] = targetParameter;
    setHyperParameters(hyperParameters);
  }

  const addHyperParameter = () => {
    setHyperParameters([
      ...hyperParameters,
      { key: '', value: '', id: hyperParameters.length + 1
    }])
  }

  const removeHyperParameter = (id: number) => {
    setHyperParameters(hyperParameters.filter(parameter => parameter.id !== id));
  }

  return (
    <Box display="flex" flexDirection="column" gap="20px" height="100%" overflow="hidden" overflowY="auto" alignItems="center" padding="10px">
      <Box
        backgroundColor="#f2f2f2"
        borderRadius="4px"
        padding="10px"
        display="flex"
        flexDirection="column"
        gap="20px"
        width="70%"
      >
        <Text as="h4" textAlign="center">Base model</Text>

        <SimpleGrid
          columns={2}
          spacing="20px"
        >
          <FormControl display="flex" justifyContent="space-between">
            <FormLabel fontSize="12px" flex="1">Name</FormLabel>

            <Input
              backgroundColor="white"
              width="70%"
              size="xs"
              placeholder='name'
              {...register('basemodelName')}
            />
          </FormControl>

          <FormControl display="flex" justifyContent="space-between">
            <FormLabel fontSize="12px" flex="1">Version</FormLabel>

            <Input
              backgroundColor="white"
              width="70%"
              size="xs"
              placeholder='version'
              {...register('basemodelVersion')}
            />
          </FormControl>
        </SimpleGrid>
      </Box>

      <OptimizerForm  />

      <SelectorForm />

      <Box
        backgroundColor="#f2f2f2"
        borderRadius="4px"
        padding="10px"
        display="flex"
        flexDirection="column"
        gap="20px"
        alignItems="center"
        width="70%"
      >
        <Text as="h4" textAlign="center">Hyperparameters</Text>

        <Box display="flex" justifyContent="space-around" width="100%">
          <Text as="label" fontSize="12px">Key</Text>
          <Text as="label" fontSize="12px">Value</Text>
        </Box>

        {hyperParameters.map((parameter, index) =>
          <FormControl key={parameter.id} display="flex" justifyContent="space-between" gap="20px">
            <Select
              backgroundColor="white"
              size="xs"
              placeholder="key"
              onChange={(event) => setHyperparameterValue(event, index, ArgsGroup.key)}
              value={parameter.key}
            >
              {
                hyperparametersOptions.map(param => <option key={param.id} value={param.value}>{param.label}</option>)
              }
            </Select>

            <Input
              backgroundColor="white"
              size="xs"
              value={parameter.value}
              placeholder="value"
              onChange={(event) => setHyperparameterValue(event, index, ArgsGroup.value)}
            />

            <Button size="sm" className="delete-button" leftIcon={<DeleteOutlineIcon fontSize="small" />} onClick={() => removeHyperParameter(parameter.id)} colorScheme='blue' />
          </FormControl>
        )}

        <Button
          size="sm"
          leftIcon={<AddIcon fontSize="small" />}
          onClick={addHyperParameter}
          colorScheme='blue'
          mr={3}
        >
          Add more
        </Button>
      </Box>
    </Box>
  )
}

export default ModelSpecForm