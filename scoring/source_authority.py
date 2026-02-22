from __future__ import annotations

from models.article import Article
from models.enums import ScoringDimension
from models.score import ScoreBreakdown


async def score_source_authority(article: Article) -> ScoreBreakdown:
    """Score based on the reliability rating of the article's source."""
    reliability = article.metadata.get("reliability", 0.5)

    return ScoreBreakdown(
        dimension=ScoringDimension.SOURCE_AUTHORITY,
        value=round(float(reliability), 3),
        reason=f"Source '{article.source_name}' reliability: {reliability}",
    )
