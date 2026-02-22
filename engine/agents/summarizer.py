from __future__ import annotations

from datetime import datetime, timezone
import json

import anthropic

from engine.agents.base import BaseAgent
from engine.llm_router import claude_cli_completion, local_completion
from engine.model_config import anthropic_model, cli_model, provider, summarizer_model, local_max_tokens, temperature
from models.article import ArticleSummary


SUMMARIZE_PROMPT = """\
You are a newsletter editor. Summarize the following article for a tech-focused newsletter.

Title: {title}
Source: {source}
Content: {content}

Respond in JSON with keys: headline (max 120 chars), summary (max 500 chars), key_points (list of 2-4 strings).
"""


class SummarizerAgent(BaseAgent):
    """Generates concise summaries using Claude or local Ollama based on config."""

    name = "summarizer"

    async def run(self) -> None:
        selected_provider = provider(self.context.settings)

        for article in self.context.curated_articles:
            try:
                summary = await self._summarize(article, selected_provider)
                self.context.summaries[article.id] = summary
            except Exception:
                self.logger.exception("Failed to summarize article %s", article.id)

        self.logger.info("Generated %d summaries", len(self.context.summaries))

    async def _summarize(self, article, selected_provider: str) -> ArticleSummary:  # noqa: ANN001
        prompt = SUMMARIZE_PROMPT.format(
            title=article.title,
            source=article.source_name,
            content=article.raw_content[:3000],
        )

        if selected_provider == "anthropic":
            client = anthropic.AsyncAnthropic()
            model = anthropic_model(self.context.settings)
            response = await client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
        elif selected_provider == "claude_cli":
            model = cli_model(self.context.settings)
            text = await claude_cli_completion(
                prompt=prompt,
                system="You are a newsletter summarizer. Return strict JSON only.",
                model=model,
            )
        else:
            model = summarizer_model(self.context.settings)
            temp = temperature(self.context.settings)
            max_tokens = local_max_tokens(self.context.settings, default=1024)
            text = await local_completion(
                model=model,
                prompt=prompt,
                system="You are a newsletter summarizer. Return strict JSON only.",
                temperature=temp,
                max_tokens=max_tokens,
            )

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                data = json.loads(text[start : end + 1])
            else:
                raise

        return ArticleSummary(
            article_id=article.id,
            headline=data.get("headline", article.title),
            summary=data.get("summary", ""),
            key_points=data.get("key_points", []),
            read_time_seconds=max(30, len(article.raw_content.split()) // 4),
            generated_at=datetime.now(timezone.utc),
        )
