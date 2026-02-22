from __future__ import annotations

from datetime import datetime

from models.article import Article
from models.enums import ScoringDimension
from models.score import ScoreBreakdown


async def score_timeliness(article: Article) -> ScoreBreakdown:
    """Score how recent the article is."""
    if not article.published_at:
        return ScoreBreakdown(
            dimension=ScoringDimension.TIMELINESS,
            value=0.5,
            reason="No publication date available",
        )

    age_hours = (datetime.utcnow() - article.published_at).total_seconds() / 3600

    if age_hours < 6:
        value = 1.0
    elif age_hours < 24:
        value = 0.8
    elif age_hours < 48:
        value = 0.6
    elif age_hours < 72:
        value = 0.4
    else:
        value = max(0.1, 0.3 - (age_hours - 72) / 500)

    return ScoreBreakdown(
        dimension=ScoringDimension.TIMELINESS,
        value=round(value, 3),
        reason=f"Article age: {age_hours:.1f} hours",
    )
