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
"""horizontal FL trainer."""

# pylint: disable=unused-import
from flame.mode.horizontal.syncfl.trainer import TAG_FETCH, TAG_UPLOAD, Trainer

# Redirect `flame.mode.horizontal.trainer.Trainer` to
# `flame.mode.horizontal.syncfl.trainer.Trainer
# This is for backward compatibility
