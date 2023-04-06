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
"""sampler provider class."""

from flame.config import DataSamplerType
from flame.object_factory import ObjectFactory
from flame.datasampler.default import DefaultDataSampler
from flame.datasampler.fedbalancer import FedBalancerDataSampler


class DataSamplerProvider(ObjectFactory):
    """DataSampler Provider."""

    def get(self, datasampler_name, **kwargs):
        """Return a datasampler for a given datasampler name."""
        return self.create(datasampler_name, **kwargs)


datasampler_provider = DataSamplerProvider()
datasampler_provider.register(DataSamplerType.DEFAULT, DefaultDataSampler)
datasampler_provider.register(DataSamplerType.FEDBALANCER, FedBalancerDataSampler)
