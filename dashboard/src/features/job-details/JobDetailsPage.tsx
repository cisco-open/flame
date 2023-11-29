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

import { Box, Button } from "@chakra-ui/react";
import { useNavigate } from "react-router-dom";
import JobTopology from "./components/job-topology/JobTopology";
import ArrowBackIosIcon from '@mui/icons-material/ArrowBackIos';

export const fitViewOptions = {
  padding: 1,
  maxZoom: 4
}

const JobDetailsPage = () => {
  const navigate = useNavigate();

  return (
    <>
      <Button marginTop="2px" leftIcon={<ArrowBackIosIcon fontSize="small" />} onClick={() => navigate('/jobs')} variant='link' size="xs">Back</Button>
      <Box width="100%" height="100%">
        <JobTopology />
      </Box>
    </>
  )
}

export default JobDetailsPage;

