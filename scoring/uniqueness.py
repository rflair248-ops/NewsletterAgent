from __future__ import annotations

from models.article import Article
from models.enums import ScoringDimension
from models.score import ScoreBreakdown


async def score_uniqueness(article: Article) -> ScoreBreakdown:
    """Estimate how unique/novel the article content is.

    This is a heuristic placeholder. A production version would compare
    against a vector store of previously seen articles.
    """
    # Heuristic: longer, more detailed content is more likely to be original
    word_count = len(article.raw_content.split())
    unique_words = len(set(article.raw_content.lower().split()))

    if word_count == 0:
        return ScoreBreakdown(
            dimension=ScoringDimension.UNIQUENESS,
            value=0.3,
            reason="No content to evaluate",
        )

    lexical_diversity = unique_words / word_count
    value = min(1.0, 0.3 + lexical_diversity * 0.7)

    return ScoreBreakdown(
        dimension=ScoringDimension.UNIQUENESS,
        value=round(value, 3),
        reason=f"Lexical diversity {lexical_diversity:.3f} ({unique_words}/{word_count})",
    )
