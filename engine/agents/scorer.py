from __future__ import annotations

from engine.agents.base import BaseAgent
from models.enums import ArticleStatus
from scoring.composite import compute_composite_score


class ScorerAgent(BaseAgent):
    """Runs every article through the five scoring dimensions."""

    name = "scorer"

    async def run(self) -> None:
        scored_count = 0
        for article in self.context.articles:
            if article.status != ArticleStatus.COLLECTED:
                continue

            score = await compute_composite_score(
                article,
                category_weights=self.context.source_config.category_weights,
                settings=self.context.settings,
            )
            self.context.scores[article.id] = score
            article.status = ArticleStatus.SCORED
            scored_count += 1

        self.logger.info("Scored %d articles", scored_count)
