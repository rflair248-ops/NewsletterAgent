from __future__ import annotations

from engine.agents.base import BaseAgent
from models.enums import ArticleStatus


class CuratorAgent(BaseAgent):
    """Selects the top N articles that pass the publication gate."""

    name = "curator"

    async def run(self) -> None:
        pipeline_cfg = self.context.settings.get("pipeline", {})
        min_relevance = pipeline_cfg.get("min_relevance_score", 0.4)
        min_quality = pipeline_cfg.get("min_quality_score", 0.5)
        max_articles = pipeline_cfg.get("max_articles_in_newsletter", 12)

        candidates = []
        for article in self.context.articles:
            if article.status != ArticleStatus.DEDUPLICATED:
                continue
            score = self.context.scores.get(article.id)
            if score and score.overall >= min_relevance and score.passed_gate:
                candidates.append((article, score))

        # Sort by overall score descending
        candidates.sort(key=lambda x: x[1].overall, reverse=True)

        selected = []
        for article, score in candidates[:max_articles]:
            article.status = ArticleStatus.SUMMARIZED  # ready for summarization
            selected.append(article)

        self.context.curated_articles = selected
        self.logger.info(
            "Curated %d articles from %d candidates",
            len(selected),
            len(candidates),
        )
