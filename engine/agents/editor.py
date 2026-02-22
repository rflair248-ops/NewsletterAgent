from __future__ import annotations

import anthropic

from engine.agents.base import BaseAgent
from models.enums import ArticleStatus


EDIT_PROMPT = """\
You are a copy editor for a tech newsletter. Apply the following fixes to the newsletter content.

Issues to fix:
{issues}

Suggestions to apply:
{suggestions}

Current content:
{content}

Return the corrected content only, no commentary.
"""


class EditorAgent(BaseAgent):
    """Applies reviewer feedback and produces the final newsletter output."""

    name = "editor"

    async def run(self) -> None:
        newsletter = self.context.newsletter
        review = self.context.review_result

        if not newsletter:
            self.logger.warning("No newsletter to edit")
            return

        # If review passed with no issues, skip editing
        if review and review.get("approved") and not review.get("issues"):
            self.logger.info("Review passed — no edits needed")
            self._render_output(newsletter)
            return

        # Apply edits via Claude
        client = anthropic.AsyncAnthropic()
        model = self.context.settings.get("llm", {}).get("model", "claude-sonnet-4-20250514")

        content = self._newsletter_to_markdown(newsletter)
        issues = review.get("issues", []) if review else []
        suggestions = review.get("suggestions", []) if review else []

        prompt = EDIT_PROMPT.format(
            issues="\n".join(f"- {i}" for i in issues) or "None",
            suggestions="\n".join(f"- {s}" for s in suggestions) or "None",
            content=content,
        )

        response = await client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        edited_markdown = response.content[0].text
        newsletter.markdown_body = edited_markdown

        self._render_output(newsletter)

        # Mark all articles as published
        for article in self.context.curated_articles:
            article.status = ArticleStatus.PUBLISHED

        self.logger.info("Editing complete — newsletter finalized")

    def _render_output(self, newsletter) -> None:  # noqa: ANN001
        if not newsletter.markdown_body:
            newsletter.markdown_body = self._newsletter_to_markdown(newsletter)

    @staticmethod
    def _newsletter_to_markdown(newsletter) -> str:  # noqa: ANN001
        lines = [f"# {newsletter.subject_line}", ""]
        for section in newsletter.sections:
            lines.append(f"## {section.title}")
            lines.append("")
            if section.intro_text:
                lines.append(section.intro_text)
                lines.append("")
            for item in section.items:
                lines.append(f"### {item.headline}")
                lines.append("")
                lines.append(item.summary)
                if item.key_points:
                    lines.append("")
                    for point in item.key_points:
                        lines.append(f"- {point}")
                lines.append("")
        return "\n".join(lines)
