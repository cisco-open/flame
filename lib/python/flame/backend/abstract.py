# Copyright 2022 Cisco Systems, Inc. and its affiliates
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


from abc import abstractmethod


class AbstractBackend:
    @abstractmethod
    async def _setup_server(self):
        pass

    @abstractmethod
    def uid(self):
        pass

    @abstractmethod
    def endpoint(self):
        pass

    @abstractmethod
    def connect(self, end_id, endpoint):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def configure(self, broker, job_id):
        pass
