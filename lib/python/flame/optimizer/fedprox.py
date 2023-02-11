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

"""FedProx optimizer"""
"""https://arxiv.org/abs/1812.06127"""

import logging
from ..common.util import (MLFramework, get_ml_framework_in_use)
from .regularizer.fedprox import FedProxRegularizer
from .fedavg import FedAvg

logger = logging.getLogger(__name__)


class FedProx(FedAvg):
    """FedProx class."""

    def __init__(self, mu):
        """Initialize FedProx instance."""
        ml_framework_in_use = get_ml_framework_in_use()

        # only support pytorch for fedprox
        if ml_framework_in_use != MLFramework.PYTORCH:
            raise NotImplementedError(
                "supported ml framework not found; "
                f"supported frameworks (for fedprox) are: {[MLFramework.PYTORCH.name.lower()]}")
        
        super().__init__()
        
        self.mu = mu
        # override parent's self.regularizer
        self.regularizer = FedProxRegularizer(self.mu)
        logger.debug("Initializing fedprox")
