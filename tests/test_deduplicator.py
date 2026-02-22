from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from engine.agents.deduplicator import DeduplicatorAgent
from memory.mem0_store import Mem0Store
from models.article import Article
from models.enums import ArticleStatus, ContentCategory
from pipeline.context import PipelineContext


class TestDeduplicatorAgent:
    def test_deduplicator_init(self, pipeline_context: PipelineContext):
        agent = DeduplicatorAgent(pipeline_context)
        assert agent.name == "deduplicator"

    @pytest.mark.asyncio
    async def test_removes_intra_run_duplicates(self, pipeline_context: PipelineContext):
        articles = [
            Article(
                id="a1",
                url="https://example.com/1",
                title="Breaking: Major AI Breakthrough Announced",
                source_name="Source A",
                category=ContentCategory.AI,
                raw_content="Content A",
                status=ArticleStatus.SCORED,
            ),
            Article(
                id="a2",
                url="https://example.com/2",
                title="Breaking: Major AI Breakthrough Announced Today",
                source_name="Source B",
                category=ContentCategory.AI,
                raw_content="Content B",
                status=ArticleStatus.SCORED,
            ),
            Article(
                id="a3",
                url="https://example.com/3",
                title="Completely Different Article About Space",
                source_name="Source C",
                category=ContentCategory.TECH,
                raw_content="Content C",
                status=ArticleStatus.SCORED,
            ),
        ]
        pipeline_context.articles = articles
        agent = DeduplicatorAgent(pipeline_context)
        await agent.run()

        kept = [a for a in articles if a.status == ArticleStatus.DEDUPLICATED]
        rejected = [a for a in articles if a.status == ArticleStatus.REJECTED]
        assert len(kept) == 2
        assert len(rejected) == 1

    @pytest.mark.asyncio
    async def test_cross_run_dedup_with_mem0(self, pipeline_context: PipelineContext):
        """When Mem0 returns a high-similarity match, the article is rejected."""
        mock_memory = MagicMock(spec=Mem0Store)
        mock_memory.enabled = True
        mock_memory.find_similar_articles = AsyncMock(
            return_value=[{"memory_id": "mem1", "score": 0.92, "memory": "Old article", "metadata": {}}]
        )
        pipeline_context.memory = mock_memory

        articles = [
            Article(
                id="x1",
                url="https://example.com/x1",
                title="Unique New Article Title",
                source_name="Source X",
                category=ContentCategory.TECH,
                raw_content="Some content",
                status=ArticleStatus.SCORED,
            ),
        ]
        pipeline_context.articles = articles
        agent = DeduplicatorAgent(pipeline_context)
        await agent.run()

        assert articles[0].status == ArticleStatus.REJECTED

    @pytest.mark.asyncio
    async def test_cross_run_low_similarity_passes(self, pipeline_context: PipelineContext):
        """When Mem0 match is below threshold, article passes dedup."""
        mock_memory = MagicMock(spec=Mem0Store)
        mock_memory.enabled = True
        mock_memory.find_similar_articles = AsyncMock(
            return_value=[{"memory_id": "mem1", "score": 0.50, "memory": "Vaguely related", "metadata": {}}]
        )
        pipeline_context.memory = mock_memory

        articles = [
            Article(
                id="y1",
                url="https://example.com/y1",
                title="Brand New Topic",
                source_name="Source Y",
                category=ContentCategory.AI,
                raw_content="Fresh content",
                status=ArticleStatus.SCORED,
            ),
        ]
        pipeline_context.articles = articles
        agent = DeduplicatorAgent(pipeline_context)
        await agent.run()

        assert articles[0].status == ArticleStatus.DEDUPLICATED

    def test_similarity_check(self, pipeline_context: PipelineContext):
        agent = DeduplicatorAgent(pipeline_context)
        assert agent._is_duplicate("Hello World News", ["hello world news today"]) is True
        assert agent._is_duplicate("Unique Title Here", ["Completely Different"]) is False
