from __future__ import annotations

import pytest

from engine.agents.curator import CuratorAgent
from models.article import Article
from models.enums import ArticleStatus, ContentCategory
from models.score import ArticleScore, ScoreBreakdown
from models.enums import ScoringDimension
from pipeline.context import PipelineContext


class TestCuratorAgent:
    def test_curator_init(self, pipeline_context: PipelineContext):
        agent = CuratorAgent(pipeline_context)
        assert agent.name == "curator"

    @pytest.mark.asyncio
    async def test_curator_selects_top_articles(self, pipeline_context: PipelineContext):
        articles = []
        for i in range(5):
            a = Article(
                id=f"cur{i}",
                url=f"https://example.com/{i}",
                title=f"Article {i}",
                source_name="Test",
                category=ContentCategory.TECH,
                raw_content=f"Content {i}",
                status=ArticleStatus.DEDUPLICATED,
            )
            articles.append(a)
            pipeline_context.scores[a.id] = ArticleScore(
                article_id=a.id,
                overall=0.9 - i * 0.1,
                passed_gate=True,
                breakdown=[
                    ScoreBreakdown(
                        dimension=ScoringDimension.RELEVANCE,
                        value=0.8,
                        reason="test",
                    )
                ],
            )

        pipeline_context.articles = articles
        agent = CuratorAgent(pipeline_context)
        await agent.run()

        assert len(pipeline_context.curated_articles) > 0
        assert len(pipeline_context.curated_articles) <= 12
