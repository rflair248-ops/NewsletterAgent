from __future__ import annotations

from datetime import datetime
import json

import anthropic

from engine.agents.base import BaseAgent
from engine.llm_router import claude_cli_completion, local_completion
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
        llm_cfg = self.context.settings.get("llm", {})
        local_cfg = llm_cfg.get("local", {})
        provider = local_cfg.get("provider", "ollama")

        for article in self.context.curated_articles:
            try:
                summary = await self._summarize(article, llm_cfg, local_cfg, provider)
                self.context.summaries[article.id] = summary
            except Exception:
                self.logger.exception("Failed to summarize article %s", article.id)

        self.logger.info("Generated %d summaries", len(self.context.summaries))

    async def _summarize(self, article, llm_cfg: dict, local_cfg: dict, provider: str) -> ArticleSummary:  # noqa: ANN001
        prompt = SUMMARIZE_PROMPT.format(
            title=article.title,
            source=article.source_name,
            content=article.raw_content[:3000],
        )

        if provider == "anthropic":
            client = anthropic.AsyncAnthropic()
            model = llm_cfg.get("model", "claude-sonnet-4-20250514")
            response = await client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
        elif provider == "claude_cli":
            model = llm_cfg.get("cli_model", "sonnet")
            text = await claude_cli_completion(
                prompt=prompt,
                system="You are a newsletter summarizer. Return strict JSON only.",
                model=model,
            )
        else:
            model = local_cfg.get("summarizer_model", "mistral-small")
            temperature = local_cfg.get("temperature", 0.3)
            max_tokens = local_cfg.get("max_tokens", 1024)
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
