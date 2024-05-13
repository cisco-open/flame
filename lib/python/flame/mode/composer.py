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
"""flame role composer."""

import logging
import concurrent.futures
from queue import Queue
from types import TracebackType
from typing import Optional, Type
from flame.mode.enums import LoopIndicator
from flame.mode.role import Role

logger = logging.getLogger(__name__)


class Composer(object):
    """Composer enables composition of tasklets."""

    def __init__(self) -> None:
        """Initialize the class."""
        # maintain tasklet chains
        self.chain = dict()
        self.reverse_chain = dict()

        self.unlinked_tasklets = dict()
        
        self.mc = Role.mc

    def __enter__(self):
        """Enter custom context."""
        ComposerContext.set_composer(self)
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        exc_traceback: Optional[TracebackType],
    ) -> bool:
        """Exit custom context."""
        ComposerContext.reset_composer()

    def get_tasklets_in_loop(self, start, end) -> set:
        """Return tasklets in a loop.

        This method returns all the tasklets in a loop including
        start and end tasklets.

        Parameters
        ----------
        start: a tasklet to start search; typically the first tasklet of a loop
        end: a tasklet to stop search; typcially the last tasklet of a loop

        Returns
        -------
        tasklets_in_loop: a set that contains tasklet objects
        """
        tasklets_in_loop = set()

        # traverse tasklets and execute them
        q = Queue()
        q.put(start)
        visited = set()
        visited.add(start)
        while not q.empty():
            tasklet = q.get()

            tasklets_in_loop.add(tasklet)

            if tasklet is end:
                break

            for child in self.chain[tasklet]:
                if child not in visited:
                    visited.add(child)
                    q.put(child)

        return tasklets_in_loop

    def run(self) -> None:
        """Execute tasklets in the chain."""
        # choose one tasklet
        tasklet = next(iter(self.chain))
        # get the first tasklet in the chain
        root = tasklet.get_root()

        # traverse tasklets and execute them
        q = Queue()
        q.put(root)
        visited = set()
        visited.add(root)
        while not q.empty():
            tasklet = q.get()

            if (len(tasklet.siblings)):
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasklet.siblings)) as executor:                    
                    # execute sibling taskelts in parallel
                    futures = [executor.submit(tasklet.do) for tasklet in [*tasklet.siblings]]

                    # execute first sibling tasklet
                    tasklet.do()

                    # wait for all tasks to complete
                    concurrent.futures.wait(futures)
            else:
                # execute single tasklet
                tasklet.do()

            if tasklet.is_continue() or (
                tasklet.is_last_in_loop() and not tasklet.is_loop_done()
            ):
                # we are here due to one of the following conditions:
                #
                # contition 1: tasklet's continue condition is met;
                #              so, we skip the remaing tasklets in the loop
                #              and go back to the start of the loop
                # condition 2: we reached the last tasklet of a loop
                #              but the loop exit condition is not met
                start, end = tasklet.loop_starter, tasklet
                tasklets_in_loop = self.get_tasklets_in_loop(start, end)

                # so, we reset the visited history
                visited = visited - tasklets_in_loop
                # and, we go back to the first tasklet
                tasklet = tasklet.loop_starter

                # put the tasklet that is at the start of loop
                q.put(tasklet)
                # now the tasklet is in the queue, we should go back
                # start from the tasklet at the beginning of loop
                continue

            elif tasklet.is_loop_done():
                # loop exit condition is met in the middle of loop
                # get the last tasklet and get out of the loop
                tasklet = tasklet.get_ender()

            # put unvisited children of a selected tasklet
            for child in self.chain[tasklet]:
                if child not in visited:
                    visited.add(child)
                    q.put(child)

        logger.debug("end of run")

    def unlink(self):
        """Unlink the chain of tasklets and save them in unlinked_tasklets.

        Also, reset chain and reverse_chain.
        """
        # empty unlinked tasklets dictionary; this initialization is imporant.
        # calling super().compose() in a child class means the consturction of
        # tasklet chain from the parent class.
        #
        # without this initialization, unlinked tasklets can be visiable beyond
        # the direct child class, which can cause a bug when a grand child
        # class attempts to chain tasklets.
        self.unlinked_tasklets = dict()

        tasklet = next(iter(self.chain))
        # get the first tasklet in the chain
        root = tasklet.get_root()

        # traverse tasklets
        q = Queue()
        q.put(root)
        while not q.empty():
            tasklet = q.get()

            self.unlinked_tasklets[tasklet.alias] = tasklet

            # put unvisited children of a selected tasklet
            for child in self.chain[tasklet]:
                q.put(child)

        for _, tasklet in self.unlinked_tasklets.items():
            tasklet.reset()

        self.chain = dict()
        self.reverse_chain = dict()

    def tasklet(self, alias: str):
        """Return an unlinked tasklet of a given alias."""
        # We don't intentionally use dict.get(); hence, KeyError can be raised.
        # The intention is to prevent returning None, which can cause an issue
        # later when tasklets are chained again.
        return self.unlinked_tasklets[alias]

    def clear_unlinked_tasklet_state(self, alias: str) -> None:
        """Clear the unlinked state of a tasklet of alias.

        The unlinked tasklet state is cleared if the taslket of a given alias
        is removed from the unlinked_taskets dictionary.
        """
        if alias in self.unlinked_tasklets:
            del self.unlinked_tasklets[alias]

    def print(self):
        """Print the chain of tasklets.

        This function is for debugging.
        """
        tasklet = next(iter(self.chain))
        # get the first tasklet in the chain
        root = tasklet.get_root()

        # traverse tasklets and print tasklet details
        q = Queue()
        q.put(root)
        while not q.empty():
            tasklet = q.get()

            print("-----")
            print(tasklet)

            # put unvisited children of a selected tasklet
            for child in self.chain[tasklet]:
                q.put(child)
        print("=====")
        print("done with printing chain")

    def get_tasklet(self, alias: str):
        """Get a tasklet from the composer by alias."""
        tasklet = next(iter(self.chain))
        root = tasklet.get_root()

        q = Queue()
        q.put(root)
        while not q.empty():
            tasklet = q.get()
            if tasklet.alias == alias:
                return tasklet
            for t in self.chain[tasklet]:
                q.put(t)
        else:
            raise ValueError(f"Tasklet with alias {alias} not found")

    def remove_tasklet(self, alias: str):
        """Remove a tasklet from the composer and stitches the chain."""
        # throw on tasklet is loop starter or ender
        tasklet = self.get_tasklet(alias)
        if tasklet.loop_state:
            raise NotImplementedError("Can't handle loop ends removal yet")
        
        tasklet = next(iter(self.chain))
        root = tasklet.get_root()

        if root.alias == alias:
            self.chain.pop(root)
            return None

        q = Queue()
        q.put(root)
        while not q.empty():
            parent_t = q.get()

            for child_t in self.chain[parent_t]:
                if child_t.alias == alias: # if child for removal
                    # stitch the chain
                    self.chain[parent_t].remove(child_t)
                    self.chain[parent_t].update(self.chain[child_t])
                    # remove the child from the chain
                    self.chain.pop(child_t)

                    self.reverse_chain = self.get_reverse_chain()

                    return None

            for child in self.chain[parent_t]:
                q.put(child)
        else:
            raise ValueError(f"Tasklet with alias {alias} not found")

    def _update_chain(self, new_order):
        self.chain = {k: self.chain[k] for k in new_order}
        self.reverse_chain = self.get_reverse_chain()

    def insert(self, alias, new_tasklet, after=False):
        """Insert a new tasklet before a given tasklet. If after is True, insert after the given tasklet.
        """

        if after:
            self._insert_after(alias, new_tasklet)
        else:
            self._insert_before(alias, new_tasklet)

    def _insert_before(self, alias, new_tasklet):
        initial_chain_order = list(self.chain)
        tasklet = next(iter(self.chain))
        root = tasklet.get_root()

        for t in self.chain:
            if t.alias == new_tasklet.alias:
                raise ValueError(f"Tasklet with alias '{new_tasklet.alias}' already exists")

        if root.alias == alias:
            self.chain[new_tasklet] = {root}
            # update the chain order
            updated_chain_order = initial_chain_order.copy()
            updated_chain_order.insert(initial_chain_order.index(root), new_tasklet)
            self._update_chain(updated_chain_order)
            
            return None

        q = Queue()
        q.put(root)
        while not q.empty():
            parent_t = q.get()

            for child_t in self.chain[parent_t]:
                if child_t.alias == alias:
                    # handle new_tasklet loop insertion
                    if child_t is child_t.loop_starter: # new_tasklet is new loop starter
                        new_tasklet.update_loop_attrs(
                            check_fn=child_t.loop_check_fn,
                            state=child_t.loop_state,
                            starter=new_tasklet,
                            ender=child_t.loop_ender
                        )
                        child_t.update_loop_attrs(state=LoopIndicator.NONE)
                    elif child_t is child_t.loop_ender:
                        new_tasklet.update_loop_attrs(
                            check_fn=child_t.loop_check_fn,
                            state=LoopIndicator.NONE,
                            starter=child_t.loop_starter,
                            ender=child_t
                        )

                    # relink the chain
                    self.chain[parent_t].remove(child_t)
                    self.chain[parent_t].add(new_tasklet)
                    # insert the new tasklet
                    self.chain[new_tasklet] = {child_t}
                    # update the chain order
                    updated_chain_order = initial_chain_order.copy()
                    updated_chain_order.insert(initial_chain_order.index(child_t), new_tasklet)
                    self._update_chain(updated_chain_order)

                    # update the loop if new_tasklet is in a loop
                    if new_tasklet.loop_state:
                        start, end = new_tasklet, new_tasklet.loop_ender
                        tasklets_in_loop = self.get_tasklets_in_loop(start, end)

                        for t in tasklets_in_loop:
                            t.update_loop_attrs(
                                starter=new_tasklet, 
                                ender=new_tasklet.loop_ender
                            )

                    return None

            for child in self.chain[parent_t]:
                q.put(child)

    def _insert_after(self, alias, new_tasklet):
        initial_chain_order = list(self.chain)
        tasklet = next(iter(self.chain))
        root = tasklet.get_root()

        for t in self.chain:
            if t.alias == new_tasklet.alias:
                raise ValueError(f"Tasklet with alias '{new_tasklet.alias}' already exists")
            
        q = Queue()
        q.put(root)
        while not q.empty():
            parent_t = q.get()

            if parent_t.alias == alias:
                # handle new_tasklet loop insertion
                if parent_t is parent_t.loop_ender: # new_tasklet is new loop ender
                    new_tasklet.update_loop_attrs(
                        check_fn=parent_t.loop_check_fn,
                        state=parent_t.loop_state,
                        starter=parent_t.loop_starter,
                        ender=new_tasklet
                    )
                    parent_t.update_loop_attrs(state=LoopIndicator.NONE)
                elif parent_t is parent_t.loop_starter:
                    new_tasklet.update_loop_attrs(
                        check_fn=parent_t.loop_check_fn,
                        state=LoopIndicator.NONE,
                        starter=parent_t.loop_ender,
                        ender=parent_t.loop_ender
                    )

                self.chain[new_tasklet] = {child_t for child_t in self.chain[parent_t]}
                self.chain[parent_t] = {new_tasklet}
                # update the chain order
                updated_chain_order = initial_chain_order.copy()
                updated_chain_order.insert(initial_chain_order.index(parent_t) + 1, new_tasklet)
                self._update_chain(updated_chain_order)

                # update the loop if new_tasklet is in a loop
                if new_tasklet.loop_state:
                    start, end = new_tasklet.loop_starter, new_tasklet.loop_ender
                    tasklets_in_loop = self.get_tasklets_in_loop(start, end)

                    for t in tasklets_in_loop:
                        t.update_loop_attrs(
                            starter=new_tasklet.loop_starter, 
                            ender=new_tasklet.loop_ender
                        )

                return None
            
            for child in self.chain[parent_t]:
                q.put(child)

    def get_reverse_chain(self):
        niahc = {k: set() for k in self.chain}
        for k, v in self.chain.items():
            for t in v:
                niahc[t].add(k)

        return niahc


class CloneComposer(object):
    """CloneComposer clones composer object."""

    def __init__(self, composer: Composer) -> None:
        """Initialize the class."""
        self.composer = composer

    def __enter__(self) -> Composer:
        """Enter custom context."""
        ComposerContext.set_composer(self.composer)

        return self.composer

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        exc_traceback: Optional[TracebackType],
    ) -> bool:
        """Exit custom context."""
        ComposerContext.reset_composer()


class ComposerContext(object):
    """ComposerContext maintains a context of composer."""

    _context_composer: Optional[Composer] = None

    @classmethod
    def get_composer(cls) -> Optional[Composer]:
        """get_composer returns a composer."""
        return cls._context_composer

    @classmethod
    def set_composer(cls, composer: Composer) -> None:
        """set_composer set a new composer."""
        cls._context_composer = composer

    @classmethod
    def reset_composer(cls) -> None:
        """reset_composer set a composer None."""
        cls._context_composer = None
