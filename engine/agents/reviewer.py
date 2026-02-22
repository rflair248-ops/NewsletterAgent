from __future__ import annotations

import anthropic
import json

from engine.agents.base import BaseAgent
from engine.llm_router import claude_cli_completion


REVIEW_PROMPT = """\
You are a senior newsletter editor. Review the following newsletter draft and provide feedback.

Check for:
1. Factual consistency between headlines and summaries
2. Tone alignment with brand voice: {tone}
3. Banned phrases: {banned}
4. Grammar and clarity issues

Newsletter sections:
{sections_text}

Respond in JSON with keys:
- approved (bool): whether the newsletter passes review
- issues (list of strings): any problems found
- suggestions (list of strings): improvement ideas
"""


class ReviewerAgent(BaseAgent):
    """Uses Claude to review the composed newsletter for quality and brand alignment."""

    name = "reviewer"

    async def run(self) -> None:
        newsletter = self.context.newsletter
        if not newsletter:
            self.logger.warning("No newsletter to review")
            return

        brand = self.context.brand_config
        voice = brand.get("voice", {})

        sections_text = self._format_sections(newsletter)

        llm_cfg = self.context.settings.get("llm", {})
        local_cfg = llm_cfg.get("local", {})
        provider = local_cfg.get("provider", "ollama")

        prompt = REVIEW_PROMPT.format(
            tone=voice.get("tone", "professional"),
            banned=", ".join(voice.get("banned_phrases", [])),
            sections_text=sections_text,
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
                system="You are a senior newsletter reviewer. Return strict JSON only.",
                model=model,
            )
        else:
            # fallback to claude-cli for review when local provider doesn't support this stage
            model = llm_cfg.get("cli_model", "sonnet")
            text = await claude_cli_completion(
                prompt=prompt,
                system="You are a senior newsletter reviewer. Return strict JSON only.",
                model=model,
            )

        try:
            review = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                review = json.loads(text[start : end + 1])
            else:
                review = {"approved": False, "issues": ["Review output was not valid JSON"], "suggestions": []}

        self.context.review_result = review
        self.logger.info(
            "Review %s — %d issues found",
            "PASSED" if review.get("approved") else "FAILED",
            len(review.get("issues", [])),
        )

    @staticmethod
    def _format_sections(newsletter) -> str:  # noqa: ANN001
        parts = []
        for section in newsletter.sections:
            parts.append(f"\n## {section.title}")
            for item in section.items:
                parts.append(f"- **{item.headline}**: {item.summary}")
        return "\n".join(parts)
