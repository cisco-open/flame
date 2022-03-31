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


"""custom abc meta."""

# source:
# https://stackoverflow.com/questions/23831510/abstract-attribute-not-property

from abc import ABCMeta as NativeABCMeta
from typing import Any, Callable, TypeVar, cast


class DummyAttribute:
    """DummyAttribute."""

    pass


R = TypeVar('R')


def abstract_attribute(obj: Callable[[Any], R] = None) -> R:
    """abstract_attribute decorator."""
    _obj = cast(Any, obj)

    if obj is None:
        _obj = DummyAttribute()

    _obj.__is_abstract_attribute__ = True

    return cast(R, _obj)


class ABCMeta(NativeABCMeta):
    """ABCMeta."""

    #
    def __call__(cls, *args, **kwargs):
        """Dunder call."""
        instance = NativeABCMeta.__call__(cls, *args, **kwargs)
        abstract_attributes = {
            name
            for name in dir(instance) if getattr(
                getattr(instance, name), '__is_abstract_attribute__', False
            )
        }

        if abstract_attributes:
            missing_attributes = ', '.join(abstract_attributes)
            raise NotImplementedError(
                f"Can't instantiate abstract class {cls.__name__} with"
                f" abstract attributes: {missing_attributes}"
            )

        return instance
