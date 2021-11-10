# Copyright (c) 2021 Cisco Systems, Inc. and its affiliates
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
"""fledge tasklet."""

from __future__ import annotations

import logging
from enum import Flag, auto
from queue import Queue

from .composer import ComposerContext

logger = logging.getLogger(__name__)


class LoopIndicator(Flag):
    """LoopIndicator is a flag class that contains loog begin and end flags."""

    NONE = 0
    BEGIN = auto()
    END = auto()


class Tasklet(object):
    """Tasklet is a class for defining a unit of work."""

    #
    def __init__(self, func, *args, loop_check_func=None) -> None:
        """Initialize the class."""
        self.func = func
        self.args = args
        self.loop_check_func = loop_check_func
        self.composer = ComposerContext.get_composer()
        self.loop_starter = None
        self.loop_state = LoopIndicator.NONE

    def __rshift__(self, other: Tasklet) -> Tasklet:
        """Set up connection."""
        if self not in self.composer.chain:
            self.composer.chain[self] = set()

        if other not in self.composer.chain:
            self.composer.chain[other] = set()

        if other.loop_starter and other.loop_starter not in self.composer.chain:
            self.composer.chain[other.loop_starter] = set()

        if self not in self.composer.reverse_chain:
            self.composer.reverse_chain[self] = set()

        if other not in self.composer.reverse_chain:
            self.composer.reverse_chain[other] = set()

        if other.loop_starter and other.loop_starter not in self.composer.reverse_chain:
            self.composer.reverse_chain[other.loop_starter] = set()

        if other.loop_state & LoopIndicator.END:
            self.composer.chain[self].add(other.loop_starter)
        else:
            self.composer.chain[self].add(other)

        if other.loop_state & LoopIndicator.END:
            self.composer.reverse_chain[other.loop_starter].add(self)
        else:
            self.composer.reverse_chain[other].add(self)

        return other

    def get_root(self) -> Tasklet:
        """get_root returns a root tasklet."""
        q = Queue()
        q.put(self)
        visited = set()
        visited.add(self)
        while not q.empty():
            root = q.get()

            for parent in self.composer.reverse_chain[root]:
                if parent not in visited:
                    visited.add(parent)
                    q.put(parent)

        return root

    def do(self) -> None:
        """Execute tasklet."""
        self.func(*self.args)

    def is_loop_done(self) -> bool:
        """Return if loop is done."""
        if not self.loop_check_func:
            return True

        return self.loop_check_func()

    def is_last_in_loop(self) -> bool:
        """Return if the tasklet is the last one in a loop."""
        return self.loop_state & LoopIndicator.END


def loop(tasklet: Tasklet) -> Tasklet:
    """Loop updates the boundary tasklets in a loop."""
    # the tasklet is the only tasklet in the loop
    if tasklet not in tasklet.composer.chain:
        tasklet.loop_starter = tasklet
        tasklet.loop_state = LoopIndicator.BEGIN | LoopIndicator.END

        tasklet.composer.chain[tasklet] = set()
        tasklet.composer.reverse_chain[tasklet] = set()

        return tasklet

    start_tasklet = tasklet.get_root()

    start_tasklet.loop_starter = start_tasklet
    start_tasklet.loop_state |= LoopIndicator.BEGIN

    tasklet.loop_starter = start_tasklet
    tasklet.loop_state |= LoopIndicator.END

    return tasklet
