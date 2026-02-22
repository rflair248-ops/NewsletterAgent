from __future__ import annotations

import pytest

from engine.agents.composer import ComposerAgent
from models.article import Article
from models.enums import ArticleStatus, ContentCategory, ScoringDimension
from models.score import ArticleScore, ScoreBreakdown
from policy.gates import evaluate_single_feature_gate
from pipeline.context import PipelineContext


class TestSingleFeatureGates:
    def test_gate_passes_with_multi_source_multi_domain(self):
        a1 = Article(id="a1", url="https://a.com/1", title="A", source_name="SourceA")
        a2 = Article(id="a2", url="https://b.com/2", title="B", source_name="SourceB")
        result = evaluate_single_feature_gate(["a1", "a2"], [a1, a2], min_cluster_size=2)
        assert result["ok"] is True

    def test_gate_fails_when_domain_diversity_missing(self):
        a1 = Article(id="a1", url="https://a.com/1", title="A", source_name="SourceA")
        a2 = Article(id="a2", url="https://a.com/2", title="B", source_name="SourceB")
        result = evaluate_single_feature_gate(["a1", "a2"], [a1, a2], min_cluster_size=2)
        assert result["ok"] is False
        assert "need_at_least_two_domains" in result["reasons"]


class TestSingleFeatureComposer:
    @pytest.mark.asyncio
    async def test_composer_single_feature_mode_generates_longform(self, pipeline_context: PipelineContext):
        pipeline_context.settings["pipeline"]["single_feature_mode"] = True
        pipeline_context.settings["pipeline"]["single_feature_top_k"] = 4
        pipeline_context.settings["pipeline"]["single_feature_min_cluster_size"] = 2

        articles = []
        for i in range(4):
            article = Article(
                id=f"sf{i}",
                url=f"https://domain{i}.com/story",
                title=f"Story {i}",
                source_name=f"Source {i}",
                category=ContentCategory.BUSINESS,
                raw_content="Market structure is shifting.",
                status=ArticleStatus.DEDUPLICATED,
            )
            articles.append(article)
            pipeline_context.scores[article.id] = ArticleScore(
                article_id=article.id,
                overall=0.9 - i * 0.05,
                passed_gate=True,
                breakdown=[
                    ScoreBreakdown(dimension=ScoringDimension.RELEVANCE, value=0.9, reason=""),
                ],
            )
        pipeline_context.articles = articles

        agent = ComposerAgent(pipeline_context)

        calls = [
            {"valid": True, "cluster_label": "AI procurement shift", "selected_article_ids": ["sf0", "sf1"], "cluster_rationale": "Shared client-demand shift"},
            {"thesis": "Law firm buyers are consolidating spend into AI-enabled managed services."},
            {"draft_markdown": "## Trigger\nX\n## Translation\nY\n## Revenue Implication\nZ\n## 3 Tactical Moves\n1. A\n2. B\n3. C\n## Risk/Blindspot\nR\n## Strategic Takeaway\nT"},
            {"headline": "AI buying behavior is redrawing legal service economics", "target_keyword": "legal AI strategy", "meta_description": "Daily legal intelligence on AI buying shifts.", "article_markdown": "# AI buying behavior is redrawing legal service economics\n\nKeyword in first paragraph legal AI strategy."},
        ]

        async def fake_llm_json(prompt: str, system: str):  # noqa: ARG001
            return calls.pop(0)

        agent._llm_json = fake_llm_json  # type: ignore[method-assign]
        await agent.run()

        assert pipeline_context.newsletter is not None
        assert pipeline_context.newsletter.metadata.get("mode") == "single_feature"
        assert "legal AI strategy" == pipeline_context.newsletter.metadata.get("target_keyword")
        assert pipeline_context.newsletter.total_articles == 2
        assert pipeline_context.newsletter.markdown_body.startswith("# AI buying behavior")

    @pytest.mark.asyncio
    async def test_composer_single_feature_blocks_when_cluster_too_small(self, pipeline_context: PipelineContext):
        pipeline_context.settings["pipeline"]["single_feature_mode"] = True
        pipeline_context.settings["pipeline"]["single_feature_top_k"] = 3
        pipeline_context.settings["pipeline"]["single_feature_min_cluster_size"] = 3

        articles = [
            Article(id="s1", url="https://a.com/1", title="A", source_name="A", raw_content="x"),
            Article(id="s2", url="https://b.com/2", title="B", source_name="B", raw_content="x"),
            Article(id="s3", url="https://c.com/3", title="C", source_name="C", raw_content="x"),
        ]
        for idx, article in enumerate(articles):
            pipeline_context.scores[article.id] = ArticleScore(
                article_id=article.id,
                overall=0.8 - idx * 0.1,
                passed_gate=True,
                breakdown=[ScoreBreakdown(dimension=ScoringDimension.RELEVANCE, value=0.9, reason="")],
            )
        pipeline_context.articles = articles

        agent = ComposerAgent(pipeline_context)

        async def fake_llm_json(prompt: str, system: str):  # noqa: ARG001
            return {"valid": True, "selected_article_ids": ["s1", "s2"], "cluster_label": "x", "cluster_rationale": "x"}

        agent._llm_json = fake_llm_json  # type: ignore[method-assign]

        with pytest.raises(ValueError, match="Single feature policy gate failed"):
            await agent.run()
