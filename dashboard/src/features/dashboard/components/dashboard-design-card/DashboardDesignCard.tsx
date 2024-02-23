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

import { Box, Text } from "@chakra-ui/react";
import { Design } from "../../../../entities/Design";
import './DashboardDesignCard.css';
import DesignDetailsPage from "../../../design-details/DesignDetailsPage";
import { ReactFlowProvider } from "reactflow";
import { useNavigate } from "react-router-dom";

interface Props {
  design?: Design | undefined;
}

const DashboardDesignCard = ({ design }: Props) => {
  const navigate = useNavigate();

  const navigateToDesignDetails = () => {
    navigate(`/design/${design?.id}`);
  }

  return (
    <Box className="dashboard-design-card" onClick={navigateToDesignDetails}>
      <Box>
        <ReactFlowProvider>
          <DesignDetailsPage externalDesignId={design?.id} />
        </ReactFlowProvider>
      </Box>

      <Box className="dashboard-design-card__bottom">
        <Text className="dashboard-design-card__bottom__name">{design?.name}</Text>
        <Text className="dashboard-design-card__bottom__description">A sample schema to demonstrate a TAG layout</Text>

        { design?.description && <Text  className="dashboard-design-card__description">{design?.description}</Text> }
      </Box>
    </Box>
  )
}

export default DashboardDesignCard