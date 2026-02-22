from __future__ import annotations

from datetime import datetime
import json

from engine.agents.base import BaseAgent
from engine.llm_router import local_completion
from models.article import ArticleSummary


SUMMARIZE_PROMPT = """\
You are a newsletter editor. Summarize the following article for a tech-focused newsletter.

Title: {title}
Source: {source}
Content: {content}

Respond in JSON with keys: headline (max 120 chars), summary (max 500 chars), key_points (list of 2-4 strings).
"""


class SummarizerAgent(BaseAgent):
    """Uses local Ollama model to generate concise summaries of curated articles."""

    name = "summarizer"

    async def run(self) -> None:
        local_cfg = self.context.settings.get("llm", {}).get("local", {})
        model = local_cfg.get("summarizer_model", "mistral-small")
        temperature = local_cfg.get("temperature", 0.3)
        max_tokens = local_cfg.get("max_tokens", 1024)

        for article in self.context.curated_articles:
            try:
                summary = await self._summarize(model, article, temperature, max_tokens)
                self.context.summaries[article.id] = summary
            except Exception:
                self.logger.exception("Failed to summarize article %s", article.id)

        self.logger.info("Generated %d summaries", len(self.context.summaries))

    async def _summarize(self, model: str, article, temperature: float, max_tokens: int) -> ArticleSummary:  # noqa: ANN001
        prompt = SUMMARIZE_PROMPT.format(
            title=article.title,
            source=article.source_name,
            content=article.raw_content[:3000],
        )

        text = await local_completion(
            model=model,
            prompt=prompt,
            system="You are a newsletter summarizer. Return strict JSON only.",
            temperature=temperature,
            max_tokens=max_tokens,
        )
        data = json.loads(text)

        return ArticleSummary(
            article_id=article.id,
            headline=data.get("headline", article.title),
            summary=data.get("summary", ""),
            key_points=data.get("key_points", []),
            read_time_seconds=max(30, len(article.raw_content.split()) // 4),
            generated_at=datetime.utcnow(),
        )
