from __future__ import annotations

from difflib import SequenceMatcher

from engine.agents.base import BaseAgent
from models.enums import ArticleStatus


class DeduplicatorAgent(BaseAgent):
    """Removes near-duplicate articles based on title similarity."""

    name = "deduplicator"

    SIMILARITY_THRESHOLD = 0.75

    async def run(self) -> None:
        scored = [a for a in self.context.articles if a.status == ArticleStatus.SCORED]
        seen_titles: list[str] = []
        rejected = 0

        for article in scored:
            if self._is_duplicate(article.title, seen_titles):
                article.status = ArticleStatus.REJECTED
                self.context.audit("dedup_reject", article_id=article.id, reason="duplicate")
                rejected += 1
            else:
                article.status = ArticleStatus.DEDUPLICATED
                seen_titles.append(article.title)

        self.logger.info(
            "Deduplication complete: %d kept, %d rejected",
            len(seen_titles),
            rejected,
        )

    def _is_duplicate(self, title: str, seen: list[str]) -> bool:
        normalized = title.lower().strip()
        for existing in seen:
            ratio = SequenceMatcher(None, normalized, existing.lower().strip()).ratio()
            if ratio >= self.SIMILARITY_THRESHOLD:
                return True
        return False
