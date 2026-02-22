from __future__ import annotations

import logging

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
