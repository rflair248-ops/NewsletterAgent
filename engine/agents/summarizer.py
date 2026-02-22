from __future__ import annotations

from datetime import datetime

import anthropic

from engine.agents.base import BaseAgent
from models.article import ArticleSummary


SUMMARIZE_PROMPT = """\
You are a newsletter editor. Summarize the following article for a tech-focused newsletter.

Title: {title}
Source: {source}
Content: {content}

Respond in JSON with keys: headline (max 120 chars), summary (max 500 chars), key_points (list of 2-4 strings).
"""


class SummarizerAgent(BaseAgent):
    """Uses Claude to generate concise summaries of curated articles."""

    name = "summarizer"

    async def run(self) -> None:
        client = anthropic.AsyncAnthropic()
        model = self.context.settings.get("llm", {}).get("model", "claude-sonnet-4-20250514")

        for article in self.context.curated_articles:
            try:
                summary = await self._summarize(client, model, article)
                self.context.summaries[article.id] = summary
            except Exception:
                self.logger.exception("Failed to summarize article %s", article.id)

        self.logger.info("Generated %d summaries", len(self.context.summaries))

    async def _summarize(self, client: anthropic.AsyncAnthropic, model: str, article) -> ArticleSummary:  # noqa: ANN001
        prompt = SUMMARIZE_PROMPT.format(
            title=article.title,
            source=article.source_name,
            content=article.raw_content[:3000],
        )

        response = await client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        import json

        text = response.content[0].text
        data = json.loads(text)

        return ArticleSummary(
            article_id=article.id,
            headline=data.get("headline", article.title),
            summary=data.get("summary", ""),
            key_points=data.get("key_points", []),
            read_time_seconds=max(30, len(article.raw_content.split()) // 4),
            generated_at=datetime.utcnow(),
        )
