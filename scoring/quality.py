from __future__ import annotations

from models.article import Article
from models.enums import ScoringDimension
from models.score import ScoreBreakdown


async def score_quality(article: Article) -> ScoreBreakdown:
    """Score article quality using heuristic signals."""
    score = 0.5

    # Penalize very short content
    word_count = len(article.raw_content.split())
    if word_count < 50:
        score -= 0.2
    elif word_count > 200:
        score += 0.15

    # Boost for having a title
    if article.title and len(article.title) > 10:
        score += 0.1

    # Source reliability
    reliability = article.metadata.get("reliability", 0.7)
    score += reliability * 0.2

    return ScoreBreakdown(
        dimension=ScoringDimension.QUALITY,
        value=round(max(0.0, min(1.0, score)), 3),
        reason=f"Word count {word_count}, reliability {reliability}",
    )
