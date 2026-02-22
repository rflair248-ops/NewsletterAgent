from __future__ import annotations

from models.article import Article
from models.enums import ScoringDimension
from models.score import ScoreBreakdown


async def score_relevance(article: Article, category_weights: dict[str, float]) -> ScoreBreakdown:
    """Score article relevance based on category weight and content signals."""
    base = category_weights.get(article.category.value, 0.5)

    # Boost for articles with substantial content
    content_length = len(article.raw_content)
    length_bonus = min(0.2, content_length / 5000)

    value = min(1.0, base * 0.7 + length_bonus + 0.1)

    return ScoreBreakdown(
        dimension=ScoringDimension.RELEVANCE,
        value=round(value, 3),
        reason=f"Category weight {base:.2f}, content length {content_length}",
    )
