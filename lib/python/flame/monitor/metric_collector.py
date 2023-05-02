# Copyright 2023 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""Metric Collector."""

import logging

logger = logging.getLogger(__name__)

class MetricCollector:
    def __init__(self):
        """Initialize Metric Collector."""
        self.state_dict = dict()
    
    def save(self, mtype, alias, value):
        """Saves key-value pair for metric."""
        key = f'{mtype}-{alias}'
        self.state_dict[key] = value
        logger.debug(f"Saving state_dict[{key}] = {value}")
    
    def get(self):
        """Returns the current metrics that were collected and clears the saved dictionary."""
        temp = self.state_dict
        self.state_dict = dict()
        return temp
    
