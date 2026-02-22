from __future__ import annotations

import pytest

feedparser = pytest.importorskip("feedparser", reason="feedparser not installed")

from engine.agents.collector import CollectorAgent  # noqa: E402
from pipeline.context import PipelineContext  # noqa: E402


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
