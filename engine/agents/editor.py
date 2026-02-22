from __future__ import annotations

import anthropic

from engine.agents.base import BaseAgent
from engine.llm_router import claude_cli_completion, local_completion
from models.enums import ArticleStatus
from policy.quality_checks import (
    append_quality_check,
    critic_scorecard,
    revise_with_fixes,
    rule_based_scorecard,
)


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
            await self._run_single_feature_quality_checks(newsletter)
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
        elif provider == "claude_cli":
            model = llm_cfg.get("cli_model", "sonnet")
            edited_markdown = await claude_cli_completion(
                prompt=prompt,
                system="You are a newsletter copy editor. Return corrected markdown only.",
                model=model,
            )
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

        await self._run_single_feature_quality_checks(newsletter)
        self._render_output(newsletter)

        # Mark all articles as published and persist to memory
        for article in self.context.curated_articles:
            article.status = ArticleStatus.PUBLISHED
            await self._remember_article(article)

        self.logger.info("Editing complete — newsletter finalized")

    async def _run_single_feature_quality_checks(self, newsletter) -> None:  # noqa: ANN001
        pipeline_cfg = self.context.settings.get("pipeline", {})
        enabled = bool(pipeline_cfg.get("single_feature_mode", False)) or (
            (newsletter.metadata or {}).get("mode") == "single_feature"
        )
        if not enabled:
            return

        if not newsletter.markdown_body:
            self._render_output(newsletter)

        keyword = (newsletter.metadata or {}).get("target_keyword", "").strip()
        source_urls = [str(a.url) for a in self.context.curated_articles if a.url]
        cli_model = self.context.settings.get("llm", {}).get("cli_model", "sonnet")

        fallback_used = False
        try:
            score = await critic_scorecard(
                markdown=newsletter.markdown_body,
                keyword=keyword,
                source_urls=source_urls,
                model=cli_model,
            )
        except Exception as exc:  # noqa: BLE001
            fallback_used = True
            self.logger.warning("Quality critic unavailable; using rule-based fallback: %s", exc)
            score = rule_based_scorecard(newsletter.markdown_body, keyword, source_urls)
            score["warning"] = "critic_unavailable_rule_based_fallback"

        append_quality_check(score, run_id=self.context.run_id)

        if score.get("pass"):
            self.logger.info("Single-feature QA passed")
            return

        fixes = [str(f) for f in score.get("actionable_fixes", []) if str(f).strip()]
        self.logger.warning("Single-feature QA failed; applying one revision pass")

        if not fallback_used:
            try:
                revised = await revise_with_fixes(
                    markdown=newsletter.markdown_body,
                    keyword=keyword,
                    fixes=fixes,
                    model=cli_model,
                )
                newsletter.markdown_body = revised
                recheck = await critic_scorecard(
                    markdown=newsletter.markdown_body,
                    keyword=keyword,
                    source_urls=source_urls,
                    model=cli_model,
                )
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Revision/recheck critic unavailable; falling back to rules: %s", exc)
                recheck = rule_based_scorecard(newsletter.markdown_body, keyword, source_urls)
                recheck["warning"] = "critic_unavailable_after_revision"
        else:
            # Deterministic fallback path: do not block publish when critic is unavailable.
            recheck = rule_based_scorecard(newsletter.markdown_body, keyword, source_urls)
            recheck["warning"] = "rule_based_recheck_only"

        append_quality_check(recheck, run_id=self.context.run_id)
        if not recheck.get("pass"):
            self.logger.warning("Single-feature QA still failing after one revision pass; proceeding with warning")

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
