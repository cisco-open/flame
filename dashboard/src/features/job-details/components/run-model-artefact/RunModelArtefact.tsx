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

import { Box, Icon, Text } from '@chakra-ui/react';
import { useContext, useEffect, useState } from 'react'
import ApiClient from '../../../../services/api-client';
import useArtifact from '../../hooks/useArtifact';
import useArtifacts from '../../hooks/useArtifacts';
import { ArtifactContext } from '../../ArtifactContext';
import { getFileStructure, getInitialFileStructure } from '../../utils';
import ArtifactTree from '../artifact-tree/ArtifactTree';
import DownloadTwoToneIcon from '@mui/icons-material/DownloadTwoTone';
import Loading from '../../../../layout/loading/Loading';
import { JobDetailsContext } from '../../JobDetailsContext';

interface Props {
  runDetails: any;
}

const RunModelArtefact = ({ runDetails }: Props) => {
  const [ artifacts, setArtifacts ] = useState<any>();
  const [ artifactFile, setArtifactFile ] = useState<any>();
  const [ selectedArtifact, setSelectedArtifact ] = useState<any>();
  const [ fileStructure, setFileStructure ] = useState<any>();
  const [ selectedFile, setSelectedFile ] = useState<any>();
  const [ mappedFileStructure, setMappedFileStructure ] = useState<any>();
  const [ artifactLoading, setArtifactLoading ] = useState<boolean>(false);
  const { data: artifact } = useArtifact({ run_uuid: runDetails?.info?.run_uuid, path: selectedArtifact?.path});
  const { data, isLoading: areArtifactsLoading } = useArtifacts({ run_uuid: runDetails?.info?.run_uuid });

  useEffect(() => {
    if (!artifact) { return; }

    if (selectedFile?.data?.name !== 'model.pth') {
      setArtifactFile(artifact);
    } else {
      setArtifactFile('You can now download this file.');
    }

    setArtifactLoading(false);
  }, [artifact]);

  useEffect(() => {
    if (!data || !data.files) {
      setMappedFileStructure(undefined);
      return;
    }

    setFileStructure(getInitialFileStructure(data))
    fetchArtifacts(data.files).then(res => { setArtifacts(res) });
  }, [data]);

  useEffect(() => {
    if (!artifacts || !fileStructure || fileStructure[0]?.children?.length) { return; }
    const structure = getFileStructure(artifacts, fileStructure);
    setMappedFileStructure(structure);
  }, [artifacts, fileStructure]);

  useEffect(() => {
    if (!selectedFile?.data) { return; }

    if (!selectedFile?.children?.length && selectedFile?.data?.isDir) {
      fetchArtifact(selectedFile).then(res => {
        const newFileStructure = [...fileStructure];
        const targetParentIndex = newFileStructure[0]?.children?.findIndex((file: any) => file.name === selectedFile?.data?.name);
        const parent = newFileStructure[0]?.children?.[targetParentIndex];
        parent.children = res.files.map((file: any, index: number) => {
          const fileName = file.path.split(`${parent.path}/`)[1];

          return {
            id: `${index}-${parent?.id}-${fileName}`,
            parentName: parent.name,
            name: fileName,
            path: file.path,
            isDir: file.is_dir
          }
        });

        setFileStructure(newFileStructure);
      });
    } else {
      if (!selectedFile?.data?.isDir) {
        setArtifactLoading(true);
      }
      setSelectedArtifact(selectedFile?.data);
    }
  }, [selectedFile]);

  const onDownloadFile = () => {
    const blob = new Blob([artifactFile], { type: 'text/plain' });

    const url = URL.createObjectURL(blob);

    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', selectedFile.data.name);

    document.body.appendChild(link);

    link.click();

    // Cleanup
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  const fetchArtifacts = async (files: any[]) => {
    const responses: any[] = [];
    const apiClient = new ApiClient('mlflow/artifacts/list', true);
    for (const file of files) {
      try {
        const response = await apiClient.getAll({ params: { run_uuid: runDetails?.info?.run_uuid, path: file?.is_dir ? file?.path : '' }});
        responses.push(response)
      } catch (error) {
        console.error('Error fetching artifacts:', error);
      }
    }

    return responses[0];
  };

  const fetchArtifact = async (file: any) => {
    const apiClient = new ApiClient('mlflow/artifacts/list', true);
    let response: any;
    try {
      response = await apiClient.getAll({ params: { run_uuid: runDetails?.info?.run_uuid, path: file?.data?.path }});
    } catch (error) {
      console.error('Error fetching artifacts:', error);
    }
    return response;
  }

  const onFileSelect = (data: any) => {
    if (!data?.data?.isDir) {
      setArtifactLoading(true);
    }
    setSelectedFile(data);
  }

  if (areArtifactsLoading) {
    return <Loading />
  }

  if (!data.files) {
    return <Box height="100%" display="flex" justifyContent="center" alignItems="center">
      <Text>No data reported.</Text>
    </Box>
  }

  return (
    <Box display="flex" gap="20px" height="100%">
      <Box width="35%">
        <ArtifactContext.Provider value={{ onFileSelect }}>
          {mappedFileStructure && <ArtifactTree data={mappedFileStructure} />}
        </ArtifactContext.Provider>
      </Box>

      <Box position="relative" width="65%">
          { artifactLoading ?
            <Loading size="md" message="Loading artifacts..."/> :
            <>
              {
                artifactFile &&
                <Icon cursor="pointer" position="absolute" top="10px" right="10px" width="40px" height="40px" zIndex="1" as={DownloadTwoToneIcon} onClick={onDownloadFile} />
              }

              {
                artifactFile ?
                  <pre style={{ fontSize: "10px"}}>
                    {artifactFile}
                  </pre> :
                  <Box height="100%" display="flex" justifyContent="center" flexDirection="column" alignItems="center">
                    <Text as="h4">Select a file to preview.</Text>
                    <Text as="p" fontSize="10px">Supported formats: text files</Text>
                  </Box>
              }
            </>
          }
      </Box>
    </Box>
  )
}

export default RunModelArtefact;