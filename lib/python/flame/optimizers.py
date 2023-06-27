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
"""optimizer provider class."""

from flame.config import OptimizerType
from flame.object_factory import ObjectFactory
from flame.optimizer.fedadagrad import FedAdaGrad
from flame.optimizer.fedadam import FedAdam
from flame.optimizer.fedavg import FedAvg
from flame.optimizer.fedbuff import FedBuff
from flame.optimizer.feddyn import FedDyn
from flame.optimizer.fedgft import FedGFT
from flame.optimizer.fedprox import FedProx
from flame.optimizer.fedyogi import FedYogi
from flame.optimizer.scaffold import Scaffold


class OptimizerProvider(ObjectFactory):
    """Optimizer Provider."""

    def get(self, optimizer_name, **kwargs):
        """Return an optimizer for a given optimizer name."""
        return self.create(optimizer_name, **kwargs)


optimizer_provider = OptimizerProvider()
optimizer_provider.register(OptimizerType.FEDAVG, FedAvg)
optimizer_provider.register(OptimizerType.FEDADAGRAD, FedAdaGrad)
optimizer_provider.register(OptimizerType.FEDADAM, FedAdam)
optimizer_provider.register(OptimizerType.FEDYOGI, FedYogi)
optimizer_provider.register(OptimizerType.FEDBUFF, FedBuff)
optimizer_provider.register(OptimizerType.FEDPROX, FedProx)
optimizer_provider.register(OptimizerType.FEDDYN, FedDyn)
optimizer_provider.register(OptimizerType.SCAFFOLD, Scaffold)
optimizer_provider.register(OptimizerType.FEDGFT, FedGFT)
