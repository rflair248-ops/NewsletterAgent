from __future__ import annotations

import anthropic

from engine.agents.base import BaseAgent
from engine.llm_router import local_completion
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

        llm_cfg = self.context.settings.get("llm", {})
        local_cfg = llm_cfg.get("local", {})
        provider = local_cfg.get("provider", "ollama")

        content = self._newsletter_to_markdown(newsletter)
        issues = review.get("issues", []) if review else []
        suggestions = review.get("suggestions", []) if review else []

        prompt = EDIT_PROMPT.format(
            issues="\n".join(f"- {i}" for i in issues) or "None",
            suggestions="\n".join(f"- {s}" for s in suggestions) or "None",
            content=content,
        )

        if provider == "anthropic":
            client = anthropic.AsyncAnthropic()
            model = llm_cfg.get("model", "claude-sonnet-4-20250514")
            response = await client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            edited_markdown = response.content[0].text
        else:
            model = local_cfg.get("editor_model", "llama3.1:8b")
            temperature = local_cfg.get("temperature", 0.3)
            max_tokens = local_cfg.get("max_tokens", 2048)
            edited_markdown = await local_completion(
                model=model,
                prompt=prompt,
                system="You are a newsletter copy editor. Return corrected markdown only.",
                temperature=temperature,
                max_tokens=max_tokens,
            )
        newsletter.markdown_body = edited_markdown

        self._render_output(newsletter)

        # Mark all articles as published and persist to memory
        for article in self.context.curated_articles:
            article.status = ArticleStatus.PUBLISHED
            await self._remember_article(article)

        self.logger.info("Editing complete — newsletter finalized")

    async def _remember_article(self, article) -> None:  # noqa: ANN001
        """Store published article in Mem0 for cross-run dedup."""
        memory = self.context.memory
        if not memory.enabled:
            return

        summary = self.context.summaries.get(article.id)
        summary_text = summary.summary if summary else ""

        await memory.store_article(
            article_id=article.id,
            title=article.title,
            source=article.source_name,
            summary=summary_text,
            run_id=self.context.run_id,
        )
        await memory.store_decision(
            run_id=self.context.run_id,
            article_id=article.id,
            decision="published",
            reason=f"Included in edition {self.context.newsletter.edition_id if self.context.newsletter else 'unknown'}",
        )

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
