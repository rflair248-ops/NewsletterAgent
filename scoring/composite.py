from __future__ import annotations

from engine.model_config import scoring_weights
from models.article import Article
from models.score import ArticleScore
from scoring.quality import score_quality
from scoring.relevance import score_relevance
from scoring.source_authority import score_source_authority
from scoring.timeliness import score_timeliness
from scoring.uniqueness import score_uniqueness

async def compute_composite_score(
    article: Article,
    category_weights: dict[str, float],
    settings: dict,
    mem0_store: object | None = None,
) -> ArticleScore:
    """Run all five scoring modules and produce a composite score."""
    breakdowns = [
        await score_relevance(article, category_weights),
        await score_quality(article),
        await score_timeliness(article),
        await score_uniqueness(article, mem0_store=mem0_store),
        await score_source_authority(article),
    ]

    dimension_weights = scoring_weights(settings)
    overall = sum(
        b.value * dimension_weights.get(b.dimension.value, 0.2)
        for b in breakdowns
    )
    overall = round(min(1.0, max(0.0, overall)), 3)

    pipeline_cfg = settings.get("pipeline", {})
    min_quality = pipeline_cfg.get("min_quality_score", 0.5)

    # Article passes gate if overall score meets threshold
    passed = overall >= min_quality

    return ArticleScore(
        article_id=article.id,
        overall=overall,
        breakdown=breakdowns,
        passed_gate=passed,
    )
