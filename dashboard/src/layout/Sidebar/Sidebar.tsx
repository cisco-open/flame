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

import { VStack, Image, Text } from '@chakra-ui/react';
import { NavLink, useNavigate } from 'react-router-dom';
import logo from '../../assets/flame-logo.png';
import './Sidebar.css';
import menuItems from '../../menu-items';

const Sidebar = () => {
  const navigate = useNavigate();
  return (
    <VStack
      className="sidebar"
      alignItems="flex-start"
      borderRight='1px solid rgba(58, 53, 65, 0.12)'
      gap="5px"
      padding="10px 5px 10px 0"
      height="100%"
      boxShadow="rgba(58, 53, 65, 0.42) 0px 4px 8px -4px"
    >
      <Image
        paddingLeft="5px"
        height='50px'
        src={logo}
        marginBottom="-5px"
        onClick={() => navigate('/')}
        cursor='pointer'
      />

      {menuItems.map(item =>
        <NavLink to={item.url} key={item.id} className="sidebar-link">
          {item.icon}

          <p className="item-title">{item.title}</p>
        </NavLink>
      )}
    </VStack>
  )
}

export default Sidebar