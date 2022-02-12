"""Analyzer abstract class."""

import logging
from typing import Any, List, Union

from ..common.typing import Metrics
from . import AbstractAnalyzer

logger = logging.getLogger(__name__)


class DummyAnalyzer(AbstractAnalyzer):
    """Dummy analyzer.

    Dummy analyzer is only for testing.
    To enable it create a yaml file (e.g., dummy.yaml) in /etc/fledge/plugin
    with the following key-valure pairs:

    class: DummyAnalyzer
    package: fledge.analyzer.dummy
    type: analyzer
    """

    def run(self,
            model: Any = None,
            dataset: Union[None, List[Any]] = None) -> Union[None, Metrics]:
        """Run analysis and return results."""
        logger.info("dummy analyzer: doing nothing")

        return None
