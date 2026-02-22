from __future__ import annotations

import anthropic

from engine.agents.base import BaseAgent


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

        client = anthropic.AsyncAnthropic()
        model = self.context.settings.get("llm", {}).get("model", "claude-sonnet-4-20250514")

        prompt = REVIEW_PROMPT.format(
            tone=voice.get("tone", "professional"),
            banned=", ".join(voice.get("banned_phrases", [])),
            sections_text=sections_text,
        )

        response = await client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        import json

        text = response.content[0].text
        review = json.loads(text)

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
