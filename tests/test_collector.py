from __future__ import annotations

import pytest

from engine.agents.collector import CollectorAgent
from pipeline.context import PipelineContext


class TestCollectorAgent:
    def test_collector_init(self, pipeline_context: PipelineContext):
        agent = CollectorAgent(pipeline_context)
        assert agent.name == "collector"
        assert agent.context is pipeline_context

    @pytest.mark.asyncio
    async def test_collector_context_manager(self, pipeline_context: PipelineContext):
        agent = CollectorAgent(pipeline_context)
        async with agent as a:
            assert a is agent
