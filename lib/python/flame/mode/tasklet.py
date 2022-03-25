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
"""flame tasklet."""

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

    def __init__(self, func, *args) -> None:
        """Initialize the class."""
        self.func = func
        self.args = args
        self.loop_check_fn = None
        self.composer = ComposerContext.get_composer()
        self.loop_starter = None
        self.loop_ender = None
        self.loop_state = LoopIndicator.NONE

    def __rshift__(self, other: Tasklet) -> Tasklet:
        """Set up connection."""
        if self not in self.composer.chain:
            self.composer.chain[self] = set()

        if other not in self.composer.chain:
            self.composer.chain[other] = set()

        # case 1: t1 >> loop(t2 >> t3)
        # if t1 is self, t3 is other; t3.loop_starter is t2
        if other.loop_starter and other.loop_starter not in self.composer.chain:
            self.composer.chain[other.loop_starter] = set()

        if self not in self.composer.reverse_chain:
            self.composer.reverse_chain[self] = set()

        if other not in self.composer.reverse_chain:
            self.composer.reverse_chain[other] = set()

        # same as case 1
        if other.loop_starter and other.loop_starter not in self.composer.reverse_chain:
            self.composer.reverse_chain[other.loop_starter] = set()

        if other.loop_state & LoopIndicator.END:
            # same as case 1
            self.composer.chain[self].add(other.loop_starter)
        else:
            self.composer.chain[self].add(other)

        if other.loop_state & LoopIndicator.END:
            # same as case 1
            self.composer.reverse_chain[other.loop_starter].add(self)
        else:
            self.composer.reverse_chain[other].add(self)

        return other

    def get_composer(self):
        """Return composer object."""
        return self.composer

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

    def get_ender(self) -> Tasklet:
        """Get last tasklet in a loop.

        If a tasklet is not in a loop, then loop_ender is None.
        In such a case, the current tasklet object is returned.
        If the current tasklet is indeed in a loop and is the last tasklet,
        it returns itself.
        Otherwise, it returns a its member variable, loop_ender.

        Returns
        -------
        tasklet: a last tasklet in a loop or an entire chain
        """
        if self.is_last_in_loop() or self.loop_ender is None:
            return self

        return self.loop_ender

    def do(self) -> None:
        """Execute tasklet."""
        self.func(*self.args)

    def is_loop_done(self) -> bool:
        """Return if loop is done."""
        if not self.loop_check_fn:
            return True

        return self.loop_check_fn()

    def is_last_in_loop(self) -> bool:
        """Return if the tasklet is the last one in a loop."""
        return self.loop_state & LoopIndicator.END


class Loop(object):
    """Loop class."""

    def __init__(self, loop_check_fn=None) -> None:
        """Initialize loop object.

        Parameters
        ----------
        loop_check_fn: a function object to check loop exit conditions
        """
        self.loop_check_fn = loop_check_fn

    def __call__(self, ender: Tasklet) -> Tasklet:
        """Configure boundaries of loop and its exit condition.

        Given an ender of a loop, the starter (i.e., the first tasklet of a loop)
        is obtained. Then, all the tasklets in the loop are obtained by using
        the two tasklets. The ender is specified for every tasklets in the loop.
        The purpose of these updates are to facilitate the traverse of tasklets.

        Parameters
        ----------
        ender: last tasklet in a loop

        Returns
        -------
        ender: last tasklet in a loop
        """
        # composer is univercially shared across tasklets
        # let's get it from ender
        composer = ender.get_composer()

        # the tasklet is the sole tasklet in a loop
        # if there are more than one tasklet in the loop,
        # ender should have been added in the chain
        # when rshift (i.e, >>) was handled
        if ender not in composer.chain:
            ender.loop_starter = ender
            ender.loop_ender = ender
            ender.loop_state = LoopIndicator.BEGIN | LoopIndicator.END
            ender.loop_check_fn = self.loop_check_fn

            composer.chain[ender] = set()
            composer.reverse_chain[ender] = set()

            return ender

        # since tasklets in a loop are not yet chained with tasklets outside
        # the loop, calling get_root() returns the first tasklet of the loop.
        starter = ender.get_root()

        starter.loop_starter = starter
        starter.loop_state |= LoopIndicator.BEGIN

        ender.loop_starter = starter
        ender.loop_state |= LoopIndicator.END

        tasklets_in_loop = composer.get_tasklets_in_loop(starter, ender)
        # for each tasklet in loop, loop_check_fn and loop_ender are updated
        for tasklet in tasklets_in_loop:
            tasklet.loop_check_fn = self.loop_check_fn
            tasklet.loop_ender = ender

        return ender
