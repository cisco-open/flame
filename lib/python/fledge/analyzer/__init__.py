"""Analyzer abstract class."""

from abc import abstractmethod
from typing import Any, List, Union

from ..common.typing import Metrics
from ..plugin import Plugin


class AbstractAnalyzer(Plugin):
    """Abstract base class for analyzer implementation."""

    def callback(self):
        """Return callback function."""
        return self.run

    @abstractmethod
    def run(self,
            model: Any = None,
            dataset: Union[None, List[Any]] = None) -> Union[None, Metrics]:
        """Run analysis and return results."""
