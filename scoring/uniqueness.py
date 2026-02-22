from __future__ import annotations

from models.article import Article
from models.enums import ScoringDimension
from models.score import ScoreBreakdown

# Threshold above which a Mem0 similarity match counts as "seen before"
SIMILARITY_SEEN_THRESHOLD = 0.85


async def score_uniqueness(
    article: Article,
    mem0_store: object | None = None,
) -> ScoreBreakdown:
    """Score how unique/novel the article is.

    Two-tier approach:
    1. If a Mem0Store is available, search for previously seen articles and
       penalise if a high-similarity match is found (cross-run dedup).
    2. Fall back to a lexical-diversity heuristic when memory is unavailable.
    """
    mem0_penalty = 0.0
    mem0_reason = ""

    if mem0_store is not None:
        try:
            from memory.mem0_store import Mem0Store

            if isinstance(mem0_store, Mem0Store) and mem0_store.enabled:
                similar = await mem0_store.find_similar_articles(
                    title=article.title,
                    content=article.raw_content[:300],
                    limit=3,
                )
                if similar:
                    best_score = max(r.get("score", 0.0) for r in similar)
                    if best_score >= SIMILARITY_SEEN_THRESHOLD:
                        mem0_penalty = 0.4
                        mem0_reason = f"Mem0 match score {best_score:.3f} (seen before). "
                    elif best_score >= 0.6:
                        mem0_penalty = 0.15
                        mem0_reason = f"Mem0 partial match {best_score:.3f}. "
        except Exception:
            pass  # graceful degradation

    # Lexical diversity heuristic
    word_count = len(article.raw_content.split())
    unique_words = len(set(article.raw_content.lower().split()))

    if word_count == 0:
        value = 0.3
        reason = "No content to evaluate"
    else:
        lexical_diversity = unique_words / word_count
        value = min(1.0, 0.3 + lexical_diversity * 0.7)
        reason = f"Lexical diversity {lexical_diversity:.3f} ({unique_words}/{word_count})"

    # Apply Mem0 penalty
    value = max(0.0, value - mem0_penalty)
    reason = mem0_reason + reason

    return ScoreBreakdown(
        dimension=ScoringDimension.UNIQUENESS,
        value=round(value, 3),
        reason=reason,
    )
