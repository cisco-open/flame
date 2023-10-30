import { Box, Checkbox, CheckboxGroup, FormControl, FormLabel, Input, Text, Textarea } from '@chakra-ui/react'
import { Channel, MappedFuncTag } from '../../../../entities/DesignDetails'
import * as yup from 'yup';
import { yupResolver } from '@hookform/resolvers/yup';
import { useForm } from 'react-hook-form';
import { useEffect, useState } from 'react';
import { getChannelPayload, getFuncTagsFromFile, getMappedFuncTags } from '../../utils';
import './ChannelDetails.css';

interface Props {
  channel: Channel;
  channels: Channel[];
  onSave: (data: any) => void;
  funcTags: MappedFuncTag[];
}

export const ChannelDetails = ({ channels, channel, funcTags, onSave }: Props) => {
  const [mappedFuncTags, setMappedFuncTags] = useState<MappedFuncTag[]>();
  const schema = yup.object().shape({
    name: yup.string().required(),
    description: yup.string(),
    isDataConsumer: yup.boolean(),
    groupBy: yup.string().required(),
  });

  const { register, getValues, setValue } = useForm({
    resolver: yupResolver(schema)
  });

  useEffect(() => {
    setValue('name', channel.name);
    setValue('description', channel.description);
    setValue('groupBy', channel.groupBy.value?.join(', ') || '');
    if (channel.funcTags && Object.keys(channel.funcTags).length === channel.pair.length ) {
      setMappedFuncTags(getMappedFuncTags(channel, funcTags, channels));
    } else {
      setMappedFuncTags(getFuncTagsFromFile([...funcTags], channel) as unknown as any);
    }
  }, [channel, channels]);

  const handleSave = () => {
    onSave(getChannelPayload(getValues(), channel, mappedFuncTags));
  }

  const onFuncTagChange = (event: any, targetRoleName: string) => {
    const updatedFuncTags = mappedFuncTags?.map(mappedTag => {
      if (mappedTag.roleName === targetRoleName) {
        return {
          ...mappedTag,
          funcTags: mappedTag.funcTags.map(t => ({
            ...t,
            selected: event.includes(t.value)
          }))
        };
      }
      return mappedTag;
    });
    setMappedFuncTags(updatedFuncTags)
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
            <Input size="xs" {...register('name')} />
          </FormControl>

          <FormControl>
            <FormLabel fontSize="12px">Description</FormLabel>
            <Textarea size="xs" {...register('description')} />
          </FormControl>

          <FormControl>
            <FormLabel fontSize="12px">Group By (comma separated values)</FormLabel>

            <Input size="xs" placeholder="Group By" {...register('groupBy')} />
          </FormControl>
          {
            mappedFuncTags &&
            mappedFuncTags.map(tag => (
              <FormControl key={tag.roleName} display="flex" flexDirection="column" gap="10px" className="channel-details-func-tag-control">
                <FormLabel margin="0" fontSize="12px"><Text as="span" fontWeight="bold">{tag.roleName}</Text> function tags</FormLabel>
                <CheckboxGroup
                  onChange={(e) => onFuncTagChange(e, tag.roleName)}
                  value={tag.funcTags.filter(tag => tag.selected).map(tag => tag.value)}
                >
                  {
                    tag.funcTags?.map(tag =>
                      <Checkbox
                        size="xs"
                        key={tag.value}
                        value={tag.value}
                        isChecked={tag.selected}
                        isDisabled={tag.disabled}
                      >
                        {tag.value}
                      </Checkbox>
                    )
                  }
                </CheckboxGroup>
              </FormControl>
            ))
          }
        </Box>
      </Box>
    </Box>
  )
}
