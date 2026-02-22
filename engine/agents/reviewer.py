from __future__ import annotations

import anthropic
import json

from engine.agents.base import BaseAgent
from engine.llm_router import claude_cli_completion
from engine.model_config import anthropic_model, cli_model, provider


REVIEW_PROMPT = """\
You are a senior newsletter editor. Review the following newsletter draft and provide feedback.

Check for:
1. Factual consistency between headlines and summaries
2. Tone alignment with brand voice: {tone}
3. Banned phrases: {banned}
4. Grammar and clarity issues

Newsletter content:
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

        sections_text = self._format_newsletter_content(newsletter)

        selected_provider = provider(self.context.settings)

        prompt = REVIEW_PROMPT.format(
            tone=voice.get("tone", "professional"),
            banned=", ".join(voice.get("banned_phrases", [])),
            sections_text=sections_text,
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
                system="You are a senior newsletter reviewer. Return strict JSON only.",
                model=model,
            )
        else:
            # fallback to claude-cli for review when local provider doesn't support this stage
            model = cli_model(self.context.settings)
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
    def _format_newsletter_content(newsletter) -> str:  # noqa: ANN001
        if newsletter.markdown_body and newsletter.markdown_body.strip():
            return newsletter.markdown_body

        parts = []
        for section in newsletter.sections:
            parts.append(f"\n## {section.title}")
            for item in section.items:
                parts.append(f"- **{item.headline}**: {item.summary}")
        return "\n".join(parts)
