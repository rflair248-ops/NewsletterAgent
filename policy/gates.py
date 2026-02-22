from __future__ import annotations

import logging
from urllib.parse import urlparse

from models.article import Article
from models.score import ArticleScore

logger = logging.getLogger(__name__)


def passes_publication_gate(
    article: Article,
    score: ArticleScore,
    min_relevance: float = 0.4,
    min_quality: float = 0.5,
) -> bool:
    """Check whether an article meets minimum thresholds for publication."""
    from models.enums import ScoringDimension

    relevance = score.dimension_value(ScoringDimension.RELEVANCE)
    quality = score.dimension_value(ScoringDimension.QUALITY)

    if relevance < min_relevance:
        logger.debug(
            "Article %s rejected: relevance %.3f < %.3f",
            article.id,
            relevance,
            min_relevance,
        )
        return False

    if quality < min_quality:
        logger.debug(
            "Article %s rejected: quality %.3f < %.3f",
            article.id,
            quality,
            min_quality,
        )
        return False

    return True


def evaluate_single_feature_gate(
    selected_ids: list[str],
    all_candidates: list[Article],
    min_cluster_size: int,
) -> dict:
    """Validate single-feature cluster quality gates."""
    reasons: list[str] = []

    by_id = {a.id: a for a in all_candidates}
    selected = [by_id[i] for i in selected_ids if i in by_id]

    if len(selected) < min_cluster_size:
        reasons.append(f"cluster_below_min_size:{len(selected)}<{min_cluster_size}")

    if len(selected) < 2:
        reasons.append("need_at_least_two_independent_sources")

    unique_sources = {a.source_name.strip().lower() for a in selected if a.source_name}
    if len(unique_sources) < 2:
        reasons.append("need_at_least_two_source_entities")

    unique_domains = {urlparse(str(a.url)).netloc.lower() for a in selected if a.url}
    if len(unique_domains) < 2:
        reasons.append("need_at_least_two_domains")

    return {"ok": len(reasons) == 0, "reasons": reasons}
