import { Box, Checkbox, FormControl, FormLabel, Text, Input, Textarea } from '@chakra-ui/react'
import { FileUploader } from 'react-drag-drop-files';
import { MappedFuncTag, Role } from '../../../../entities/DesignDetails';
import './RoleDetails.css';
import * as yup from 'yup';
import { yupResolver } from "@hookform/resolvers/yup";
import { useForm } from 'react-hook-form';
import { useEffect } from 'react';
import { FUNC_TAGS_MAPPING } from '../../constants';

interface Props {
  role: Role;
  setFileData: (file: any) => void;
  onSave: (data: any) => void;
  setFuncTags: (data: {roleName: string; funcTags: any }) => void;
  fileNames: { node: string, name: string }[];
}

const fileTypes = ["py"];

const RoleDetails = ({ role, setFileData, onSave, setFuncTags, fileNames }: Props) => {
  useEffect(() => {
    setValue('name', role.name);
    setValue('description', role.description);
    setValue('isDataConsumer', role.isDataConsumer);
    setValue('replica', role.replica);
  }, [role]);

  const handleChange = (file: FileList) => {
    const fileReader = new FileReader();
    fileReader.readAsText(file[0], "text/plain");
    fileReader.onload = e => {
      const result = e?.target?.result as string;
      const funcTags = FUNC_TAGS_MAPPING.find(tag => result.includes(tag.fileValue))?.funcTags;

      if (!funcTags) { return; }
      setFuncTags({ roleName: role.name, funcTags: funcTags.map(tag => ({ value: tag, selected: false, disabled: false })) });
    };
    setFileData({
      file: file[0],
      roleName: role.name
    });
  };

  const schema = yup.object().shape({
    name: yup.string().required(),
    description: yup.string(),
    isDataConsumer: yup.boolean(),
    replica: yup.number(),
  });

  const { register, reset, getValues, setValue } = useForm({
    resolver: yupResolver(schema)
  });

  const handleSave = () => {
    const formValue = getValues();
    onSave({
      ...role,
      ...formValue,
      replica: formValue.replica ? +formValue.replica : 0,
      previousName: role.name,
    });
  }

  return (
    <Box className="side-panel-overlay" onClick={handleSave}>
      <Box
        flex="1"
        boxShadow="rgba(58, 53, 65, 0.42) -4px -4px 8px -4px"
        height="100vh"
        overflowY="auto"
        padding="10px"
        position="absolute"
        right="0"
        top="0"
        backgroundColor="white"
        zIndex="2"
      >
        <Box display="flex" flexDirection="column" gap="20px" width="30vw" zIndex="2" onClick={(e) => e.stopPropagation()}>
          <FormControl>
            <FormLabel fontSize="12px">Name</FormLabel>
            <Input
              size="xs"
              placeholder='ID'
              { ...register('name') }
            />
          </FormControl>


          <FormControl>
            <FormLabel fontSize="12px">Replica</FormLabel>
            <Input
              size="xs"
              placeholder='Replica'
              { ...register('replica') }
            />
          </FormControl>

          <FormControl>
            <FormLabel fontSize="12px">Description</FormLabel>
            <Textarea
              size="xs"
              { ...register('description') }
            />
          </FormControl>

          <FormControl display="flex" alignItems="center" gap="10px">
            <Checkbox
              { ...register('isDataConsumer') }
            />
            <FormLabel margin="0" fontSize="12px">Is data consumer</FormLabel>
          </FormControl>

          <Text as="label">Group association</Text>

          {role.groupAssociation?.map((group: any, index: number) => {
            return (
              <Box key={index} display="flex" flexDirection="column" gap="10px">
                <Text fontSize="12px">Group: {index + 1}</Text>

                {
                  Object.keys(group).map((key: any) =>
                    <FormControl display="flex" gap="10px" key={key}>
                      <Input size="xs" readOnly placeholder='ID' value={key} />
                      <Input size="xs" readOnly placeholder='ID' value={group[key]} />
                    </FormControl>
                  )
                }
              </Box>
            )
          })}

          <Text>Code file: {fileNames?.find(file => file.node === role.name)?.name || 'N/A'}</Text>

          <FileUploader multiple label={fileNames?.find(file => file.node === role.name) ? 'Change code file' : 'Add code file'} handleChange={handleChange} name="file" types={fileTypes} />
        </Box>
      </Box>
    </Box>
  )
}

export default RoleDetails