from __future__ import annotations

from difflib import SequenceMatcher

from engine.agents.base import BaseAgent
from models.enums import ArticleStatus


class DeduplicatorAgent(BaseAgent):
    """Removes near-duplicate articles based on title similarity.

    Two-level dedup:
    1. **Intra-run** — SequenceMatcher on titles within the current batch.
    2. **Cross-run** — Mem0 similarity search against articles from previous
       pipeline runs (when memory is enabled).
    """

    name = "deduplicator"

    SIMILARITY_THRESHOLD = 0.75
    MEM0_SIMILARITY_THRESHOLD = 0.85

    async def run(self) -> None:
        scored = [a for a in self.context.articles if a.status == ArticleStatus.SCORED]
        seen_titles: list[str] = []
        rejected = 0

        for article in scored:
            # Intra-run dedup (title similarity within this batch)
            if self._is_duplicate(article.title, seen_titles):
                article.status = ArticleStatus.REJECTED
                self.context.audit("dedup_reject", article_id=article.id, reason="intra_run_duplicate")
                rejected += 1
                continue

            # Cross-run dedup via Mem0
            if await self._seen_in_memory(article):
                article.status = ArticleStatus.REJECTED
                self.context.audit("dedup_reject", article_id=article.id, reason="cross_run_duplicate")
                rejected += 1
                continue

            article.status = ArticleStatus.DEDUPLICATED
            seen_titles.append(article.title)

        self.logger.info(
            "Deduplication complete: %d kept, %d rejected",
            len(seen_titles),
            rejected,
        )

    async def _seen_in_memory(self, article) -> bool:  # noqa: ANN001
        """Check Mem0 for a high-similarity match from a previous run."""
        memory = self.context.memory
        if not memory.enabled:
            return False

        try:
            similar = await memory.find_similar_articles(
                title=article.title,
                content=article.raw_content[:300],
                limit=1,
                run_id=self.context.run_id,
            )
            if similar:
                best = similar[0]
                if best.get("score", 0.0) >= self.MEM0_SIMILARITY_THRESHOLD:
                    self.logger.debug(
                        "Cross-run duplicate: %s matched memory %s (score=%.3f)",
                        article.id,
                        best.get("memory_id"),
                        best["score"],
                    )
                    return True
        except Exception:
            self.logger.debug("Mem0 cross-run check failed for %s", article.id, exc_info=True)

        return False

    def _is_duplicate(self, title: str, seen: list[str]) -> bool:
        normalized = title.lower().strip()
        for existing in seen:
            ratio = SequenceMatcher(None, normalized, existing.lower().strip()).ratio()
            if ratio >= self.SIMILARITY_THRESHOLD:
                return True
        return False
