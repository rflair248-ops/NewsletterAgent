from __future__ import annotations

import pytest

from engine.agents.scorer import ScorerAgent
from models.article import Article
from models.enums import ArticleStatus
from pipeline.context import PipelineContext


class TestScorerAgent:
    def test_scorer_init(self, pipeline_context: PipelineContext):
        agent = ScorerAgent(pipeline_context)
        assert agent.name == "scorer"

    @pytest.mark.asyncio
    async def test_scorer_scores_collected_articles(
        self,
        pipeline_context: PipelineContext,
        sample_articles: list[Article],
    ):
        pipeline_context.articles = sample_articles
        agent = ScorerAgent(pipeline_context)
        await agent.run()

        assert len(pipeline_context.scores) == len(sample_articles)
        for article in sample_articles:
            assert article.status == ArticleStatus.SCORED
            score = pipeline_context.scores[article.id]
            assert 0.0 <= score.overall <= 1.0
