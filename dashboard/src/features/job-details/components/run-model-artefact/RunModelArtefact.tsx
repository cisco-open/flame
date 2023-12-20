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
import { useEffect, useState } from 'react'
import ApiClient from '../../../../services/api-client';
import useArtifact from '../../hooks/useArtifact';
import useArtifacts from '../../hooks/useArtifacts';
import { ArtifactContext } from '../../ArtifactContext';
import ArtifactTree from '../artifact-tree/ArtifactTree';
import DownloadTwoToneIcon from '@mui/icons-material/DownloadTwoTone';
import Loading from '../../../../layout/loading/Loading';

interface Props {
  runDetails: any;
}

const MAX_FILE_SIZE_FOR_PREVIEW = 5000;

const RunModelArtefact = ({ runDetails }: Props) => {
  const [ responses, setResponses ] = useState<any>([]);
  const [ artifactFile, setArtifactFile ] = useState<any>();
  const [ artifactPreview, setArtifactPreview ] = useState<any>();
  const [ selectedArtifact, setSelectedArtifact ] = useState<any>();
  const [ selectedFile, setSelectedFile ] = useState<any>();
  const [ mappedFileStructure, setMappedFileStructure ] = useState<any>();
  const [ artifactLoading, setArtifactLoading ] = useState<boolean>(false);
  const { data: artifact } = useArtifact({ run_uuid: runDetails?.info?.run_uuid, path: selectedArtifact?.path});
  const { data, isLoading: areArtifactsLoading } = useArtifacts({ run_uuid: runDetails?.info?.run_uuid });

  useEffect(() => {
    if (!artifact) { return; }

    if (selectedFile?.data?.file_size < MAX_FILE_SIZE_FOR_PREVIEW) {
      setArtifactPreview(artifact)
    } else {
      setArtifactPreview('You can now download this file.');
    }
    setArtifactFile(artifact);

    setArtifactLoading(false);
  }, [artifact]);

  useEffect(() => {
    if (!data || !data.files) {
      setMappedFileStructure(undefined);
      return;
    }
    const responses = [...data.files];
    setResponses((prevValue: any) => [...prevValue, ...responses]);
    fetchArtifacts(responses.map((r: any) => {
      const index = r.path.lastIndexOf('/');
      const fileName = r.path.slice(index + 1);

      return {
        ...r,
        isDir: r.is_dir,
        id: `${fileName}`,
        name: fileName,
      }
    }));
  }, [data]);

  useEffect(() => {
    if (!selectedFile?.data || selectedFile?.data?.is_dir) { return; }

    setArtifactLoading(true);
    setSelectedArtifact(selectedFile?.data);
  }, [selectedFile]);

  const onDownloadFile = () => {
    const blob = new Blob([artifactFile]);

    const url = URL.createObjectURL(blob);

    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', selectedFile.data.name);

    document.body.appendChild(link);

    link.click();

    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  useEffect(() => {
    setMappedFileStructure([...responses.filter((data: any) =>
      data.is_dir &&
      data.children?.length &&
      data.path.split('/').length === 1 &&
      !!data.id
    )]);
  }, [responses])

  const fetchArtifacts = async (files: any[]) => {
    const apiClient = new ApiClient('mlflow/artifacts/list', true);

    if (!files?.length) { return; }

    for (let i = 0; i < files.length; i++) {
      if (files[i].is_dir) {
        try {
          const response: any = await apiClient.getAll({ params: { run_uuid: runDetails?.info?.run_uuid, path: files[i]?.is_dir ? files[i]?.path : '' }});
          files[i]['children'] = response.files.map(((f: any) => {
            const index = f.path.lastIndexOf('/');
            const fileName = f.path.slice(index + 1);
            return {
              ...f,
              name: fileName,
              id: f.path
            }
          }));
        } catch (error) {
          console.error('Error fetching artifacts:', error);
        }
      }
      fetchArtifacts(files[i].children);
    }
    setResponses((prevValue: any) => [...prevValue, ...files]);
  };

  const onFileSelect = (data: any) => {
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
                    {artifactPreview}
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