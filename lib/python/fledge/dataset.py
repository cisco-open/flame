# Copyright (c) 2022 Cisco Systems, Inc. and its affiliates
# All rights reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Dataset class."""

from numpy import ndarray


class Dataset(object):
    """Dataset class."""

    def __init__(self,
                 x: ndarray = None,
                 y: ndarray = None,
                 dataloader=None) -> None:
        """Initialize Dataset.

        A dataset is created either by using x and y or by using dataloader.
        Both x and y must have valid or be None.
        At least one of (x, y) or dataloader must not be None.

        Parameters
        ----------
        x: numpy array for data
        y: numpy array for labels (or targets in pytorch term)
        dataloader: an instance of torch.utils.data.DataLoader or
                    pytorch's custom dataloader

        Returns
        -------
        None
        """
        if x is None and y is None and dataloader is None:
            raise ValueError("x, y and dataloader are None")

        if dataloader:
            # in case pytorch's dataloader, convert tensor to numpy array
            self.x = next(iter(dataloader))[0].numpy()  # data
            self.y = next(iter(dataloader))[1].numpy()  # targets (or labels)
        else:
            if x is None or y is None:
                raise ValueError(f"x (={x}) and y (={y}) must not be None")

            self.x = x
            self.y = y

    def get(self) -> tuple[ndarray, ndarray]:
        """Return x (data) and y (targets).

        Returns
        -------
        x: numpy array for data
        y: numpy array for labels (or targets in pytorch term)
        """
        return self.x, self.y
