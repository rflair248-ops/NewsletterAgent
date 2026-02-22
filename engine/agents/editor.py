from __future__ import annotations

import os
from datetime import datetime, timezone

import anthropic

from engine.agents.base import BaseAgent
from engine.llm_router import claude_cli_completion, local_completion
from engine.model_config import anthropic_model, cli_model, editor_model, local_max_tokens, provider, temperature
from models.enums import ArticleStatus
from policy.quality_checks import (
    append_quality_check,
    critic_scorecard,
    revise_with_fixes,
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
            self._append_sources_footer(newsletter)
            self._upload_to_google_sheet(newsletter)
            for article in self.context.curated_articles:
                article.status = ArticleStatus.PUBLISHED
                await self._remember_article(article)
            return

        selected_provider = provider(self.context.settings)

        content = self._newsletter_to_markdown(newsletter)
        issues = review.get("issues", []) if review else []
        suggestions = review.get("suggestions", []) if review else []

        prompt = EDIT_PROMPT.format(
            issues="\n".join(f"- {i}" for i in issues) or "None",
            suggestions="\n".join(f"- {s}" for s in suggestions) or "None",
            content=content,
        )

        if selected_provider == "anthropic":
            client = anthropic.AsyncAnthropic()
            model = anthropic_model(self.context.settings)
            response = await client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            edited_markdown = response.content[0].text
        elif selected_provider == "claude_cli":
            model = cli_model(self.context.settings)
            edited_markdown = await claude_cli_completion(
                prompt=prompt,
                system="You are a newsletter copy editor. Return corrected markdown only.",
                model=model,
            )
        else:
            model = editor_model(self.context.settings)
            temp = temperature(self.context.settings)
            max_tokens = local_max_tokens(self.context.settings)
            edited_markdown = await local_completion(
                model=model,
                prompt=prompt,
                system="You are a newsletter copy editor. Return corrected markdown only.",
                temperature=temp,
                max_tokens=max_tokens,
            )
        newsletter.markdown_body = edited_markdown

        await self._run_single_feature_quality_checks(newsletter)
        self._render_output(newsletter)
        self._append_sources_footer(newsletter)
        self._upload_to_google_sheet(newsletter)

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
        cli_model_value = cli_model(self.context.settings)

        try:
            score = await critic_scorecard(
                markdown=newsletter.markdown_body,
                keyword=keyword,
                source_urls=source_urls,
                model=cli_model_value,
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Quality critic unavailable; blocking publish: %s", exc)
            raise RuntimeError("Single-feature publish blocked: Claude critic unavailable") from exc

        append_quality_check(score, run_id=self.context.run_id)

        if score.get("pass"):
            self.logger.info("Single-feature QA passed")
            return

        fixes = [str(f) for f in score.get("actionable_fixes", []) if str(f).strip()]
        self.logger.warning("Single-feature QA failed; applying one revision pass")

        try:
            revised = await revise_with_fixes(
                markdown=newsletter.markdown_body,
                keyword=keyword,
                fixes=fixes,
                model=cli_model_value,
            )
            newsletter.markdown_body = revised
            recheck = await critic_scorecard(
                markdown=newsletter.markdown_body,
                keyword=keyword,
                source_urls=source_urls,
                model=cli_model_value,
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Revision/recheck critic unavailable; blocking publish: %s", exc)
            raise RuntimeError("Single-feature publish blocked: Claude critic unavailable during recheck") from exc

        append_quality_check(recheck, run_id=self.context.run_id)
        if not recheck.get("pass"):
            self.logger.warning("Single-feature QA still failing after one revision pass; proceeding with warning")

    def _append_sources_footer(self, newsletter) -> None:  # noqa: ANN001
        """Append canonical sources footer to final markdown output."""
        if not newsletter.markdown_body:
            return

        seen: set[str] = set()
        lines = ["", "---", "## Sources", ""]
        for article in self.context.curated_articles:
            url = str(article.url)
            if url in seen:
                continue
            seen.add(url)
            lines.append(f"- {article.source_name}: [{article.title}]({url})")

        if len(seen) == 0:
            return

        if "## Sources" not in newsletter.markdown_body:
            newsletter.markdown_body = newsletter.markdown_body.rstrip() + "\n" + "\n".join(lines) + "\n"

    def _upload_to_google_sheet(self, newsletter) -> None:  # noqa: ANN001
        """Upload generated doc to Google Sheet when configured."""
        output_cfg = self.context.settings.get("output", {})
        sheets_cfg = output_cfg.get("google_sheets", {})
        if not sheets_cfg.get("enabled", False):
            return

        sheet_id = sheets_cfg.get("sheet_id") or os.getenv("GOOGLE_SHEET_ID")
        worksheet_name = sheets_cfg.get("worksheet", "articles")
        creds_path = sheets_cfg.get("service_account_json") or os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")

        if not sheet_id or not creds_path:
            self.logger.warning("Google Sheets upload enabled but missing sheet_id or service account path")
            return

        try:
            import gspread

            gc = gspread.service_account(filename=creds_path)
            sh = gc.open_by_key(sheet_id)
            try:
                ws = sh.worksheet(worksheet_name)
            except Exception:  # noqa: BLE001
                ws = sh.add_worksheet(title=worksheet_name, rows=1000, cols=12)
                ws.append_row(["timestamp_utc", "run_id", "edition_id", "subject_line", "article_markdown", "source_urls"])

            source_urls = ", ".join(sorted({str(a.url) for a in self.context.curated_articles if a.url}))
            ws.append_row(
                [
                    datetime.now(timezone.utc).isoformat(),
                    self.context.run_id,
                    newsletter.edition_id,
                    newsletter.subject_line,
                    newsletter.markdown_body,
                    source_urls,
                ]
            )
            self.logger.info("Uploaded newsletter to Google Sheet worksheet=%s", worksheet_name)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Google Sheets upload failed: %s", exc)

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
