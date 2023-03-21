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
from torch.utils.data import Dataset


class PyTorchDatasetWithIndex(Dataset):
    """
    PyTorch Dataset class that returns tuple of
    data, target, index.
    """

    def __init__(self, dataset: Dataset) -> None:
        """Initialize with existing Dataset instance."""

        self.dataset = dataset

    def __getitem__(self, index):
        """Adds index at the return tuple item."""

        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        """Returns the length of the dataset."""

        return len(self.dataset)
