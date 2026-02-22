from __future__ import annotations

import pytest

from engine.agents.deduplicator import DeduplicatorAgent
from models.article import Article
from models.enums import ArticleStatus, ContentCategory
from pipeline.context import PipelineContext


class TestDeduplicatorAgent:
    def test_deduplicator_init(self, pipeline_context: PipelineContext):
        agent = DeduplicatorAgent(pipeline_context)
        assert agent.name == "deduplicator"

    @pytest.mark.asyncio
    async def test_removes_duplicates(self, pipeline_context: PipelineContext):
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

    def test_similarity_check(self, pipeline_context: PipelineContext):
        agent = DeduplicatorAgent(pipeline_context)
        assert agent._is_duplicate("Hello World News", ["hello world news today"]) is True
        assert agent._is_duplicate("Unique Title Here", ["Completely Different"]) is False
