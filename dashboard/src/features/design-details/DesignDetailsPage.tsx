/**
 * Copyright 2023 Cisco Systems, Inc. and its affiliates
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import { Box, Button, ListItem, Popover, PopoverBody, PopoverContent, PopoverTrigger, Select, Text, UnorderedList, useDisclosure } from '@chakra-ui/react';
import { useNavigate, useParams } from 'react-router-dom';
import ArrowBackIosIcon from '@mui/icons-material/ArrowBackIos';
import ReactFlow, { Background, NodeChange, applyNodeChanges, useEdgesState, addEdge, useReactFlow } from 'reactflow';
import ErrorOutlineOutlinedIcon from '@mui/icons-material/ErrorOutlineOutlined';
import React, { useCallback, useEffect, useRef, useState } from 'react';
import 'reactflow/dist/style.css';
import { Channel, MappedFuncTag, Role, Schema } from '../../entities/DesignDetails';
import { ChannelDetails } from './components/channel-details/ChannelDetails';
import RoleDetails from './components/role-details/RoleDetails';
import useDesign from './hooks/useDesign';
import {
  mapNodes,
  mapEdges,
  getUpdatedNodes,
  getUpdatedEdges,
  getDefaultNode,
  updateRolesByEdges,
  getUpdatedEdgesByRole,
  addEdgeWithDefaultChannel,
  createCodeFileData,
  getSchemaPayload,
  createDesignCodeZip,
  getSchemaValidity,
  getErrorMessages
} from './utils';
import '../../components/custom-node/customNode.css';
import './DesignDetails.css';
import { nodeTypes, edgeTypes, fitViewOptions, defaultEdgeOptions, connectionLineStyle, FUNC_TAGS_MAPPING } from './constants';

import "filepond/dist/filepond.min.css";
import useSchema from './hooks/useSchemas';
import { COLORS, LOGGEDIN_USER } from '../../constants';
import useCode from '../../hooks/useCode';
import CustomConnectionLine from '../../components/custom-connection-line/CustomConnectionLine';
import ExpandedTopology from './components/expanded-topology/ExpandedTopology';
import ConfirmationDialog from '../../components/confirmation-dialog/ConfirmationDialog';
import './animations.css';
import { FaEllipsisVertical } from 'react-icons/fa6';
import DeleteOutlineOutlinedIcon from '@mui/icons-material/DeleteOutlineOutlined';
import EditOutlinedIcon from '@mui/icons-material/EditOutlined';
import useDesigns from '../design/hooks/useDesigns';
import FullscreenExitTwoToneIcon from '@mui/icons-material/FullscreenExitTwoTone';
import FullscreenTwoToneIcon from '@mui/icons-material/FullscreenTwoTone';
import AddBoxTwoToneIcon from '@mui/icons-material/AddBoxTwoTone';
import RestartAltTwoToneIcon from '@mui/icons-material/RestartAltTwoTone';
import DesignFormModal from '../design/components/DesignFormModal';
import { DesignForm } from '../design/types';
import { DesignContext } from '../design/DesignContext';

export interface SchemaValidity {
  multipleDataConsumerRoles: boolean;
  noGroupOnEachChannel: boolean;
  codeFileMissing: boolean;
  noDataConsumerRoles: boolean;
  allRolesNotConnected: boolean;
  duplicateNames: string[];
}

const schemaFiles = [
  {
    id: 1,
    name: 'Two tier schema',
    path: '/2-tier.json',
  },
  {
    id: 2,
    name: 'Hierarchical FL schema',
    path: '/hier-fl.json'
  },
  {
    id: 3,
    name: 'Distributed schema',
    path: '/distributed.json'
  }
];

interface Props {
  externalDesignId?: string;
}

const DesignDetailsPage = ({ externalDesignId }: Props) => {
  const jsZip = require("jszip");
  const navigate = useNavigate();
  const { id } = useParams();
  const { data: design, isLoading } = useDesign(id || externalDesignId || '');
  const { updateMutation, deleteMutation } = useSchema(id || '')
  const { pushFileMutation, deleteCodeMutation } = useCode(id || '');
  const [ designSchema, setDesignSchema ] = useState<any>();
  const [ isMenuOpened, setIsMenuOpened ] = useState<boolean>(false);
  const [ selectedDesignTemplate, setSelectedDesignTemplate ] = useState<any>();
  const [ nodes, setNodes ] = useState<any>([]);
  const [ funcTags, setFuncTags ] = useState<MappedFuncTag[]>([]);
  const [ errorMessages, setErrorMessages ] = useState<string[]>([]);
  const [ schemaValidity, setSchemaValidity ] = useState<SchemaValidity>({
    multipleDataConsumerRoles: false,
    noGroupOnEachChannel: false,
    codeFileMissing: false,
    noDataConsumerRoles: false,
    allRolesNotConnected: false,
    duplicateNames: [],
  });

  const { isOpen: isForceDeleteOpen, onOpen: onForceDeleteOpen, onClose: onForceDeleteClose } = useDisclosure();
  const [fileNames, setFileNames] = useState<any>([]);
  const [designInEdit, setDesignInEdit] = useState<any>();
  const [fileData, setFileData] = useState<any[]>([]);
  const [channel, setChannel] = useState<Channel | undefined>(undefined);
  const [role, setRole] = useState<Role | undefined>(undefined);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [areEdgesUpdated, setAreEdgesUpdated] = useState<boolean>();
  const [displayExpandedTopology, setDisplayExpandedTopology] = useState<boolean>(false);
  const { fitView } = useReactFlow();
  const { isOpen: isDeleteSchemaOpen, onOpen: onDeleteSchemaOpen, onClose: onDeleteSchemaClose } = useDisclosure();
  const { isOpen: isDeleteDesignOpen, onOpen: onDeleteDesignOpen, onClose: onDeleteDesignClose } = useDisclosure();
  const { isOpen: isEditDesignOpen, onOpen: onEditDesignOpen, onClose: onEditDesignClose } = useDisclosure();
  const {
    data: designs,
    deleteMutation: deleteDesignMutation,
    updateMutation: updateDesignMutation,
    forceDeleteMutation,
  } = useDesigns({ designInEdit, navigate, onClose: onEditDesignClose, onForceDeleteOpen });
  const { isOpen: isPopoverOpen, onToggle: onPopoverToggle, onClose: onPopoverClose } = useDisclosure();

  const isOpenReference = useRef<boolean>();
  isOpenReference.current = isPopoverOpen;

  useEffect(() => {
    window.addEventListener('click', closeMenuOnClick);

    return () => {
      window.removeEventListener('click', closeMenuOnClick);
    }
  }, [])

  useEffect(() => {
    setDesignSchema(design?.schema);
  }, [design]);

  useEffect(() => {
    checkSchemaValidity(nodes, edges, fileNames);
  }, [nodes, edges, fileNames]);

  useEffect(() => {
    setErrorMessages(getErrorMessages(schemaValidity));
  }, [schemaValidity])

  useEffect(() => {
    if (designSchema?.roles && id) {
      loadRoleFiles();
    }
  }, [id, designSchema]);

  useEffect(() => {
    const nodes = getNodes(designSchema || undefined, externalDesignId);
    fitViewOptions.nodes = nodes?.map(node => ({ id: node.id }));
    setNodes(nodes);
    setEdges(getEdges(designSchema || undefined, externalDesignId));
    setTimeout(() => fitView(), 10);
  }, [designSchema, externalDesignId]);

  const checkSchemaValidity = (nodes: any, edges: any, fileNames: any) => {
    setSchemaValidity(getSchemaValidity(nodes, edges, fileNames));
  }

  const closeMenuOnClick = () => {
    if(isOpenReference.current) {
      onPopoverClose();
    }
  }

  const loadRoleFiles = async () => {
    const xmlHttp = new XMLHttpRequest();
    xmlHttp.responseType = 'arraybuffer';
    xmlHttp.open('GET', `api/users/${LOGGEDIN_USER.name}/designs/${id}/code`)
    xmlHttp.overrideMimeType('text/plain; charset=x-user-defined');
    xmlHttp.send();

    xmlHttp.onload = () => {
      if (xmlHttp.status === 200) {
        jsZip.loadAsync(xmlHttp.response).then(async (zip: any) => {
          const result = await Promise.all(
            Object.keys(zip.files).map(async (filename, index) => {
              const file = await zip.files[filename].async('string');
              const blobFile = await zip.files[filename].async('blob');
              const fileObject = new File([blobFile], Object.keys(zip.files)[index].split('/')[1]);
              const funcTags = FUNC_TAGS_MAPPING.find(tag => file.includes(tag.fileValue))?.funcTags.map(tag => ({
                value: tag,
                selected: false,
                disabled: false,
              }));
              return { roleName: filename.split('/')[0], funcTags, file: fileObject };
            })
          );
          setFuncTags(result.map(entry => ({ roleName: entry.roleName, funcTags: entry.funcTags })) as MappedFuncTag[]);

          const names = Object.keys(zip.files).map(name => ({
            node: name.split('/')[0],
            name: name.split('/')[1],
            file: zip.file(name.split('/')[1])
          }));
          setFileData(result.map(({ roleName, file }) => ({ file, roleName })));
          setFileNames(names);
        });
      }
    }
  }

  const onNodesChange = (changes: any) => {
    if (externalDesignId) { return; }

    setNodes((nds: any) => applyNodeChanges(changes, nds) as any);
  };

  const onEdgeClick = (event: React.MouseEvent, edge: any) => {
    if (externalDesignId) { return; }

    event.stopPropagation();
    const index = edges.findIndex((e: any) => e.id === edge.id);
    setChannel(undefined);
    setChannel({ ...edge.channel, index });
    setRole(undefined);
  }
  const onNodeClick = (event: React.MouseEvent, node: any) => {
    if (externalDesignId) { return; }

    onPopoverClose();
    event.stopPropagation();
    const index = nodes.findIndex((n: any) => n.id === node.id);
    setRole(undefined);
    setRole({ ...node.role, index });
    setChannel(undefined);
  }

  const onPaneClick = useCallback(
    (event?: React.MouseEvent) => {
      event?.stopPropagation();
      onPopoverClose();
      setChannel(undefined);
      setRole(undefined);
    },
    []
  );

  const onConnect = useCallback((params: any) => {
    setEdges((eds) => addEdge(addEdgeWithDefaultChannel(params, (eds || [])), (eds || [])));
  }, [setEdges]);

  const addNode = () => {
    setNodes([...(nodes || []), getDefaultNode(nodes?.length || 0)]);
  }

  const handleTemplateChange = async (event: any) => {
    const targetTemplate = schemaFiles.find(file => file.id === +event.target.value);
    try {
      const response = await fetch(targetTemplate?.path || '');
      const data = await response.json();
      setDesignSchema(data);
    } catch (error) {
      console.error('Error fetching JSON:', error);
    }
  }

  const handleRoleSave = (role: Role) => {
    setFuncTags([...funcTags.map(tag => ({
      ...tag,
      roleName: tag.roleName === role.previousName ? role.name : tag.roleName
    }))]);
    setFileData([...fileData.map(data => ({
      ...data,
      roleName: data.roleName === role.previousName ? role.name : data.roleName
    }))]);
    setFileNames([...fileNames?.map((name: any) => ({
      ...name,
      node: name.node === role.previousName ? role.name : name.node
    }))]);
    setNodes(getUpdatedNodes(role, nodes));
    setEdges(getUpdatedEdgesByRole(role, edges));
    onPaneClick();
  }

  const channelSave = useCallback((channel: Channel) => {
    setEdges(getUpdatedEdges(channel, edges) as unknown as any);
    setAreEdgesUpdated(true);
    onPaneClick();
    setTimeout(() => setAreEdgesUpdated(false), 100)
  }, [edges]);

  useEffect(() => {
    if (areEdgesUpdated) {
      setNodes(updateRolesByEdges(edges, nodes));
    }
  }, [areEdgesUpdated]);

  const handleDesignCodeSave = () => {
    createDesignCodeZip(fileData, design?.id).generateAsync({ type: 'blob' }).then((file: any) => {
      pushFileMutation.mutate({
        fileName: design?.id,
        fileData: file,
      })
    });
  };

  const handleFiles = (data: any) => {
    const codeFileData = createCodeFileData(fileData, data);
    setFileData(codeFileData);
    setFileNames(codeFileData.map(data => ({
      node: data.roleName,
      name: data.file.name,
      file: data.file,
    })));
  }

  const handleSchemaSave = () => {
    if (fileData?.length) {
      handleDesignCodeSave();
    }
    updateMutation.mutate(getSchemaPayload(designSchema, nodes, edges, design?.id));
  };

  const getNodes = (schema: Schema | undefined, externalDesignId?: string) => schema ? mapNodes(schema, externalDesignId) : [];
  const getEdges = (schema: Schema | undefined, externalDesignId?: string) => schema ? mapEdges(schema, externalDesignId) : [];

  const toggleExpandedTopology = () => {
    setDisplayExpandedTopology(!displayExpandedTopology);
  }

  const onSchemaDelete = () => {
    deleteMutation.mutate();
    deleteCodeMutation.mutate();
    setNodes([]);
    setEdges([]);
    onDeleteSchemaClose();
  }

  const handleNewFuncTags = (data: any) => {
    setFuncTags([...funcTags.filter(tag => tag.roleName !== data.roleName), data])
  }

  const handleChannelDelete = (id: string) => {
    const schema = {...designSchema};
    schema.channels = schema?.channels?.filter((channel: any) => channel.name !== id);
    const filteredEdges = edges.filter((edge: any) => edge?.channel?.name !== id);
    setEdges(filteredEdges);
    setDesignSchema(schema);
    onPaneClick();
  }

  const handleRoleDelete = (id: string) => {
    const schema = {...designSchema};
    schema.roles = schema?.roles?.filter((role: any) => role.name !== id);
    const filteredNodes = nodes.filter((node: any) => node?.role?.name !== id);
    setNodes(filteredNodes);
    setDesignSchema(schema);
    onPaneClick();
  }

  const onEditDesignClicked = (event: React.MouseEvent) => {
    setDesignInEdit(designs?.find(d => d.id === id));
    onEditDesignOpen();
  }

  const onDeleteDesignClicked = (event: React.MouseEvent) => {
    onDeleteDesignOpen();
  }

  const handleDesignDelete = () => {
    deleteDesignMutation.mutate(id);
  }

  const handleEditDesign = (data: DesignForm) => {
    updateDesignMutation.mutate({ ...data });
  };

  if (isLoading) {
    return <p>Loading...</p>
  }

  const handleForceDelete = () => {
    forceDeleteMutation.mutate({ id, queryParams: '?forceDelete=true' } );
    onForceDeleteClose();
  }

  return (
    <DesignContext.Provider value={{ designInEdit }}>
      <Box display="flex" flexDirection="column" height="100%" overflow="hidden" className="design-details">
        {
          !externalDesignId &&
          <Box display="flex" alignItems="center" height="50px" position="relative">
            <Button marginTop="2px" leftIcon={<ArrowBackIosIcon fontSize="small" />} onClick={() => navigate('/design')} variant='link' colorScheme="primary" size="xs">Designs</Button>

            <Text alignSelf="center" flex="1" textAlign="center" textTransform="uppercase" as="h2" fontWeight="bold">{design?.name}</Text>

            <Box display="flex" justifyContent="flex-end" gap="20px" alignItems="center">
              <Select
                size="xs"
                placeholder='Select design template'
                backgroundColor={COLORS.white}
                value={selectedDesignTemplate}
                onChange={(event) => handleTemplateChange(event)}
              >
                {schemaFiles?.map(file =>
                  <option key={file.id} value={file.id}>{file.name}</option>
                )}
              </Select>

              <Button isDisabled={!!errorMessages?.length} onClick={handleSchemaSave} flexShrink="0" variant="solid" size="xs" colorScheme="primary">Save Schema</Button>
            </Box>
          </Box>
        }

        <Box display="flex" height="calc(100% - 50px)" bgColor="white" borderRadius="10px" zIndex="1">
          <Box flex="3" position="relative" className={externalDesignId ? 'external-usage' : ''}>
            {
              !!errorMessages?.length && !externalDesignId &&
              <Box className="design-details__errors">
                <Popover>
                  <PopoverTrigger>
                    <ErrorOutlineOutlinedIcon className="error-pulse design-details__error-trigger" color="error"/>
                  </PopoverTrigger>
                  <PopoverContent>
                    <PopoverBody>
                      <UnorderedList>
                        {
                          !nodes?.length && !edges?.length ? <ListItem fontSize="12px">Invalid design schema</ListItem> :
                          errorMessages.map(message => <ListItem fontSize="12px" key={message}>{message}</ListItem>)
                        }
                      </UnorderedList>
                    </PopoverBody>
                  </PopoverContent>
                </Popover>
              </Box>
            }

            {
              !externalDesignId &&
              <Box position="absolute" zIndex="1" top="5px" right="5px" display="flex" flexDirection="column" alignItems="flex-end" gap="20px">
                <Popover isOpen={isPopoverOpen}>
                  <PopoverTrigger>
                    <Button onClick={(event: React.MouseEvent) => { event.stopPropagation(); onPopoverToggle()}} className='menu-button' leftIcon={<FaEllipsisVertical />} />
                  </PopoverTrigger>

                  <PopoverContent onClick={onPopoverClose}>
                    <PopoverBody className='design-details-popover-body'>
                      { !errorMessages?.length &&
                        !!nodes?.length &&
                        !!edges?.length &&
                        <Box className="design-details-menu-item" onClick={toggleExpandedTopology}>
                          {
                            displayExpandedTopology ?
                            <FullscreenExitTwoToneIcon fontSize="small"/> :
                            <FullscreenTwoToneIcon fontSize="small"/>
                          }

                          { displayExpandedTopology ? 'Collapse' : 'Expand' }
                        </Box>
                      }

                      { !displayExpandedTopology &&
                        <Box className="design-details-menu-item" onClick={addNode}>
                          <AddBoxTwoToneIcon fontSize="small"/>

                          Add Role
                        </Box>
                      }

                      {
                        !!designSchema?.roles?.length &&
                        <Box className="design-details-menu-item" onClick={onDeleteSchemaOpen}>
                          <RestartAltTwoToneIcon fontSize="small"/>

                          Reset
                        </Box>
                      }

                      <Box className="design-details-menu-item" onClick={(event) => onEditDesignClicked(event)}>
                        <EditOutlinedIcon fontSize="small"/>

                        Edit Design
                      </Box>

                      <Box className="design-details-menu-item" onClick={(event) => onDeleteDesignClicked(event)}>
                        <DeleteOutlineOutlinedIcon fontSize="small"/>

                        Delete Design
                      </Box>
                    </PopoverBody>
                  </PopoverContent>
                </Popover>
              </Box>
            }


            { displayExpandedTopology ?
              <ExpandedTopology nodes={nodes}/> :
              <ReactFlow
                nodes={nodes}
                edges={edges}
                edgeTypes={edgeTypes}
                nodeTypes={nodeTypes}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                onEdgeClick={onEdgeClick}
                onPaneClick={onPaneClick}
                onNodeClick={onNodeClick}
                onConnect={onConnect}
                defaultEdgeOptions={defaultEdgeOptions}
                connectionLineComponent={CustomConnectionLine as unknown as any}
                connectionLineStyle={connectionLineStyle}
                fitView
                className={ externalDesignId ? 'design-details__react-flow--disabled' : ''}
              >
                { !externalDesignId && <Background /> }
              </ReactFlow>
            }
          </Box>

          {
            (role || channel) &&
              <>
                {
                  role &&
                  <RoleDetails
                    role={role}
                    setFileData={handleFiles}
                    onSave={(data) => handleRoleSave(data)}
                    onDelete={handleRoleDelete}
                    setFuncTags={(data) => handleNewFuncTags(data)}
                    fileNames={fileNames}
                  />
                }
                {
                  channel &&
                  <ChannelDetails
                    channel={channel}
                    channels={edges?.map((edge: any) => edge.channel)}
                    funcTags={funcTags}
                    onSave={(data) => channelSave(data)}
                    onDelete={handleChannelDelete}
                  />
                }
              </>
          }
        </Box>

        {
          isDeleteSchemaOpen &&
          <ConfirmationDialog
            actionButtonLabel={'Delete'}
            message={'Are sure you want to reset design schema?'}
            buttonColorScheme={'red'}
            isOpen={isDeleteSchemaOpen}
            onClose={onDeleteSchemaClose}
            onAction={onSchemaDelete}
          />
        }

        {
          isDeleteDesignOpen &&
          <ConfirmationDialog
            actionButtonLabel={'Delete'}
            message={'Are sure you want to delete this design?'}
            buttonColorScheme={'red'}
            isOpen={isDeleteDesignOpen}
            onClose={onDeleteDesignClose}
            onAction={handleDesignDelete}
          />
        }

        {
          isForceDeleteOpen &&
          <ConfirmationDialog
            actionButtonLabel={'Delete'}
            message="This design is used in a job, are you sure you want to delete this anyway?"
            buttonColorScheme={'red'}
            isOpen={isForceDeleteOpen}
            onClose={onForceDeleteClose}
            onAction={handleForceDelete}
          />
        }

        { true && <DesignFormModal isOpen={isEditDesignOpen} onClose={onEditDesignClose} onSave={(data: DesignForm) => handleEditDesign(data)} /> }
      </Box>
    </DesignContext.Provider>
  )
}

export default DesignDetailsPage;
