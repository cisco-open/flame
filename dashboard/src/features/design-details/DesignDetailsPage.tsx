import { Box, Button, ListItem, Popover, PopoverBody, PopoverContent, PopoverTrigger, Text, UnorderedList, useDisclosure } from '@chakra-ui/react';
import { useNavigate, useParams } from 'react-router-dom';
import ArrowBackIosIcon from '@mui/icons-material/ArrowBackIos';
import ReactFlow, { Background, NodeChange, applyNodeChanges, useEdgesState, addEdge, useReactFlow } from 'reactflow';
import ErrorOutlineOutlinedIcon from '@mui/icons-material/ErrorOutlineOutlined';
import React, { useCallback, useEffect, useState } from 'react';
import 'reactflow/dist/style.css';
import { Channel, Role, Schema } from '../../entities/DesignDetails';
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
import { LOGGEDIN_USER } from '../../constants';
import useCode from '../../hooks/useCode';
import CustomConnectionLine from '../../components/custom-connection-line/CustomConnectionLine';
import { saveAs } from 'file-saver';
import ExpandedTopology from './components/expanded-topology/ExpandedTopology';
import ConfirmationDialog from '../../components/confirmation-dialog/ConfirmationDialog';
import './animations.css';

export interface SchemaValidity {
  multipleDataConsumerRoles: boolean;
  noGroupOnEachChannel: boolean;
  codeFileMissing: boolean;
  noDataConsumerRoles: boolean;
  allRolesNotConnected: boolean;
  duplicateNames: string[];
}

const DesignDetailsPage = () => {
  const jsZip = require("jszip");
  const navigate = useNavigate();
  const { id } = useParams();
  const { data: design, isLoading } = useDesign(id || '');
  const { updateMutation, deleteMutation } = useSchema(id || '')
  const { pushFileMutation, deleteCodeMutation } = useCode(id || '');
  const [ designSchema, setDesignSchema ] = useState<Schema | undefined>()
  const [ nodes, setNodes ] = useState<any>([]);
  const [ funcTags, setFuncTags ] = useState<any>([]);
  const [ errorMessages, setErrorMessages ] = useState<string[]>([]);
  const [ schemaValidity, setSchemaValidity ] = useState<SchemaValidity>({
    multipleDataConsumerRoles: false,
    noGroupOnEachChannel: false,
    codeFileMissing: false,
    noDataConsumerRoles: false,
    allRolesNotConnected: false,
    duplicateNames: [],
  });

  const [fileNames, setFileNames] = useState<any>();
  const [fileData, setFileData] = useState<any[]>([]);
  const [channel, setChannel] = useState<Channel | undefined>(undefined);
  const [role, setRole] = useState<Role | undefined>(undefined);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [areEdgesUpdated, setAreEdgesUpdated] = useState<boolean>();
  const [displayExpandedTopology, setDisplayExpandedTopology] = useState<boolean>(false);
  const { fitView } = useReactFlow();
  const { isOpen, onOpen, onClose } = useDisclosure();


  useEffect(() => {
    setDesignSchema(design?.schema);
  }, [design]);

  useEffect(() => {
    checkSchemaValidity(nodes, edges, fileNames);
    console.log(nodes, edges);
  }, [nodes, edges, fileNames]);

  useEffect(() => {
    setErrorMessages(getErrorMessages(schemaValidity));
  }, [schemaValidity])

  useEffect(() => {
    if (designSchema?.roles) {
      loadRoleFiles();
    }
  }, [id, designSchema]);

  useEffect(() => {
    console.log(designSchema);
    const nodes = getNodes(designSchema || undefined);
    // if (!nodes) { return; }
    fitViewOptions.nodes = nodes?.map(node => ({ id: node.id }));
    setNodes(nodes);
    setEdges(getEdges(designSchema || undefined));

    window.requestAnimationFrame(() => {
      fitView();
    });
  }, [designSchema]);

  const checkSchemaValidity = (nodes: any, edges: any, fileNames: any) => {
    setSchemaValidity(getSchemaValidity(nodes, edges, fileNames));
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
            Object.keys(zip.files).map(async (filename) => {
              const file = await zip.files[filename].async('string');
              const funcTags = FUNC_TAGS_MAPPING.find(tag => file.includes(tag.fileValue))?.funcTags.map(tag => ({
                value: tag,
                selected: false,
                disabled: false,
              }));
              return { roleName: filename.split('/')[0], funcTags };
            })
          );
          setFuncTags(result);

          const names = Object.keys(zip.files).map(name => ({
            node: name.split('/')[0],
            name: name.split('/')[1],
            file: zip.file(name.split('/')[1])
          }));
          setFileNames(names);
        });
      }
    }
  }

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => setNodes((nds: any) => applyNodeChanges(changes, nds) as any),
    []
  );

  const onEdgeClick = (event: React.MouseEvent, edge: any) => {
    event.stopPropagation();
    const index = edges.findIndex((e: any) => e.id === edge.id);
    setChannel(undefined);
    setChannel({ ...edge.channel, index });
    setRole(undefined);
  }
  const onNodeClick = (event: React.MouseEvent, node: any) => {
    event.stopPropagation();
    const index = nodes.findIndex((n: any) => n.id === node.id);
    setRole(undefined);
    setRole({ ...node.role, index });
    setChannel(undefined);
  }

  const onPaneClick = useCallback(
    (event?: React.MouseEvent) => {
      event?.stopPropagation();

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

  const handleTemplateChange = (event: any) => {
    const fileReader = new FileReader();
    fileReader.readAsText(event?.target?.files[0], "application/json");
    fileReader.onload = e => {
      onPaneClick();
      setFileNames(null);
      setDesignSchema(JSON.parse(e?.target?.result as string))
    };
  }

  const handleRoleSave = (role: Role) => {
    setNodes(getUpdatedNodes(role, nodes));
    setEdges(getUpdatedEdgesByRole(role, edges));
    onPaneClick();
  }

  const channelSave = useCallback((channel: Channel) => {
    setEdges(getUpdatedEdges(channel, edges));
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
    setFileNames(codeFileData.map(data => data.roleName));
  }

  const handleSchemaSave = () => {
    if (fileData?.length) {
      handleDesignCodeSave();
    }
    updateMutation.mutate(getSchemaPayload(designSchema, nodes, edges, design?.id));
  };

  const getNodes = (schema: Schema | undefined) => schema ? mapNodes(schema) : [];
  const getEdges = (schema: Schema | undefined) => schema ? mapEdges(schema) : [];

  const toggleExpandedTopology = () => {
    setDisplayExpandedTopology(!displayExpandedTopology);
  }

  const openResetConfirmation = () => {
    onOpen();
  }

  const handleClose = () => {
    onClose();
  }

  const onDelete = () => {
    deleteMutation.mutate();
    deleteCodeMutation.mutate();
    setNodes([]);
    setEdges([]);
    onClose();
  }

  if (isLoading) {
    return <p>Loading...</p>
  }

  return (
    <Box display="flex" flexDirection="column" height="100%" overflow="hidden">
      <Box display="flex" alignItems="center" height="50px" position="relative">
        <Button marginTop="2px" leftIcon={<ArrowBackIosIcon fontSize="small" />} onClick={() => navigate('/design')} variant='link' size="xs">Back</Button>

        <Text alignSelf="center" flex="1" textAlign="center" textTransform="uppercase" as="h2">{design?.name}</Text>

        <Box display="flex" justifyContent="flex-end" gap="20px" alignItems="center">
          <label className="file-input" htmlFor="file-input">Choose template</label>
          <input id="file-input" style={{ 'width': '60%', 'display': 'none' }} type="file" onChange={handleTemplateChange} />

          <Button isDisabled={!!errorMessages?.length} onClick={handleSchemaSave} variant="solid" size="xs" colorScheme="teal">Save Schema</Button>
        </Box>
      </Box>

      <Box display="flex" position="relative" height="calc(100% - 50px)">
        {
          !!errorMessages?.length &&
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

        <Box flex="3" position="relative">
          {
            !errorMessages?.length &&
            !!nodes?.length &&
            !!edges?.length &&
            <Button
              position="absolute"
              zIndex="1"
              top="20px"
              right="90px"
              variant='outline'
              colorScheme="teal"
              onClick={toggleExpandedTopology}
              size="xs"
            >
              { displayExpandedTopology ? 'Collapse' : 'Expand' }
            </Button>
          }

          { !displayExpandedTopology &&
            <Button
              position="absolute"
              zIndex="1"
              top="20px"
              right="5px"
              variant='outline'
              colorScheme="teal"
              onClick={addNode} size="xs"
            >
              Add Role
            </Button>
          }

          {
            !!designSchema?.roles &&
            <Button
              position="absolute"
              zIndex="1"
              top="50px"
              right="5px"
              variant='outline'
              colorScheme="red"
              onClick={openResetConfirmation}
              size="xs"
            >
              Delete
            </Button>
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
            >
              <Background />
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
                  setFuncTags={(data) => setFuncTags([...funcTags, data])}
                  fileNames={fileNames}
                />
              }
              {
                channel &&
                <ChannelDetails
                  channel={channel}
                  channels={edges.map((edge: any) => edge.channel)}
                  funcTags={funcTags}
                  onSave={(data) => channelSave(data)}
                />
              }
            </>
        }
      </Box>

      <ConfirmationDialog
        actionButtonLabel={'Delete'}
        message={'Are sure you want to reset design schema?'}
        buttonColorScheme={'red'}
        isOpen={isOpen}
        onClose={handleClose}
        onAction={onDelete}
      />
    </Box>
  )
}

export default DesignDetailsPage;
