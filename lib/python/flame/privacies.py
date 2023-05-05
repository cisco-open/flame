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
"""privacy provider class."""

from flame.config import PrivacyType
from flame.object_factory import ObjectFactory
from flame.privacy.default import DefaultPrivacy
from flame.privacy.differential_privacy import DifferentialPrivacy


class PrivacyProvider(ObjectFactory):
    """Privacy Provider."""

    def get(self, privacy_name, **kwargs):
        """Return a privacy policy class for a given name."""
        return self.create(privacy_name, **kwargs)


privacy_provider = PrivacyProvider()
privacy_provider.register(PrivacyType.DEFAULT, DefaultPrivacy)
privacy_provider.register(PrivacyType.DP, DifferentialPrivacy)
