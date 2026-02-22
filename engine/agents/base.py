from __future__ import annotations

import abc
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.context import PipelineContext

logger = logging.getLogger(__name__)


class BaseAgent(abc.ABC):
    """Abstract base class for all pipeline agents."""

    name: str = "base"

    def __init__(self, context: PipelineContext) -> None:
        self.context = context
        self.logger = logging.getLogger(f"agent.{self.name}")

    @abc.abstractmethod
    async def run(self) -> None:
        """Execute this agent's stage of the pipeline."""

    async def __aenter__(self) -> BaseAgent:
        self.logger.info("Starting %s agent", self.name)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        self.logger.info("Finished %s agent", self.name)
