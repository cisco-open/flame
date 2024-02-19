/**
 * Copyright 2024 Cisco Systems, Inc. and its affiliates
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

import { Job } from '../../../../entities/Job';
import { List, ListItem, Text } from '@chakra-ui/layout';
import './DashboardList.css';
import { JOB_STATE } from '../../../../constants';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import { JOB_STATE_COLOR } from '../../constants';
import { Box } from '@chakra-ui/react';
import { format } from 'date-fns';
import { useNavigate } from 'react-router-dom';

interface Props {
  data: Job[] | undefined;
}

const DashboardList = ({ data }: Props) => {
  const navigate = useNavigate();
  return (
    <Box display="flex" flexDirection="column" gap="20px;" alignItems="center">
      <Text>Recently added jobs</Text>
      <List spacing={1} padding="20px" width="100%">
        {
          data?.map(({ name, state, id, createdAt }) => <ListItem key={id} className="list-item" onClick={() => navigate(`/jobs/${id}`)}>
            <Text fontSize="14px">{ name }</Text>

            <Box display="flex" alignItems="center" gap="20px">
              {
                state === JOB_STATE.failed ?
                  <ErrorIcon sx={{ color: JOB_STATE_COLOR[state].borderColor }} fontSize="inherit"/> :
                  <CheckCircleIcon sx={{ color: JOB_STATE_COLOR[state].borderColor }} fontSize="inherit"/>
              }
              <Text fontSize="14px">{ state }</Text>
            </Box>

            <Text fontSize="14px">Created: { format(new Date(createdAt), 'L/d/yyyy') }</Text>
          </ListItem>)
        }

      </List>
    </Box>
  )
}

export default DashboardList