from __future__ import annotations

from datetime import datetime, timezone
import re
import json
from typing import Any
from urllib.parse import urlparse

import anthropic

from engine.agents.base import BaseAgent
from engine.llm_router import claude_cli_completion, local_completion
from engine.model_config import anthropic_model, cli_model, provider, summarizer_model, temperature
from models.enums import ArticleStatus, SectionType
from models.newsletter import Newsletter, NewsletterSection
from policy.gates import evaluate_single_feature_gate


STAGE_1_CLUSTERING_PROMPT = """\
You are a strategic intelligence clustering analyst for law firm partner/operators.
Given top-scored signals, group them by thematic overlap.

Signals JSON:
{signals_json}

Rules:
1) Cluster by shared structural shift (market, client demand, operating model, regulation, tech stack).
2) Pick the single strongest cluster by combined score strength and strategic coherence.
3) The chosen cluster must include at least {min_cluster_size} items.
4) If no cluster meets threshold, return valid=false.

Return strict JSON:
{{
  "valid": true/false,
  "reason": "string",
  "cluster_label": "string",
  "selected_article_ids": ["id1", "id2"],
  "cluster_rationale": "string"
}}
"""

STAGE_2_THESIS_PROMPT = """\
You are a strategy editor for law firm partner/operators.
Using the selected cluster, produce ONE declarative thesis sentence describing a structural shift.

Cluster context JSON:
{cluster_json}

Return strict JSON:
{{"thesis": "One declarative sentence."}}
"""

STAGE_3_SYNTHESIS_PROMPT = """\
You are writing one synthesis-heavy strategic intelligence article for law firm partner/operators.

Thesis: {thesis}
Sources JSON:
{sources_json}

Constraints:
- Use inline source evidence naturally inside analysis. Do NOT write standalone per-source summaries.
- Use this 6-part framework with H2s:
  1) Trigger
  2) Translation
  3) Revenue Implication
  4) Tactical Moves (3-5 numbered moves)
  5) Risk/Blindspot
  6) Strategic Takeaway

Return strict JSON:
{{
  "draft_markdown": "full markdown body with H2 headings and inline source links/citations"
}}
"""

STAGE_4_SEO_PROMPT = """\
You are an SEO editor. Refine this strategic article while preserving argument quality.

Draft markdown:
{draft_markdown}

Requirements:
- Add declarative H1
- Ensure H2 flow: What happened -> Why it matters -> What to do
- Integrate a primary keyword naturally in the first 100 words
- Use numbered tactical subheads (example: '3 Tactical Moves')
- Add contextual internal-link hooks (placeholder links acceptable)
- Produce a compelling meta description
- Include target keyword
- Close with forward-looking positioning, not a recap

Return strict JSON:
{{
  "headline": "H1 title without markdown prefix",
  "target_keyword": "keyword phrase",
  "meta_description": "<=160 chars",
  "article_markdown": "final markdown including H1/H2 flow"
}}
"""


def _sanitize_untrusted_text(text: str, max_len: int = 1200) -> str:
    stripped = re.sub(r"<[^>]+>", " ", text or "")
    stripped = re.sub(r"\s+", " ", stripped).strip()
    return stripped[:max_len]


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Invalid {field_name}: expected string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"Invalid {field_name}: empty")
    return cleaned


def _require_non_empty_markdown(value: Any, field_name: str) -> str:
    markdown = _require_non_empty_string(value, field_name)
    if len(markdown.split()) < 6:
        raise ValueError(f"Invalid {field_name}: too short")
    return markdown


class ComposerAgent(BaseAgent):
    """Assembles section assignments and summaries into a Newsletter object."""

    name = "composer"

    async def run(self) -> None:
        pipeline_cfg = self.context.settings.get("pipeline", {})
        if pipeline_cfg.get("single_feature_mode", False):
            await self._compose_single_feature()
            return
        self._compose_digest()

    def _compose_digest(self) -> None:
        sections: list[NewsletterSection] = []

        section_order = [
            SectionType.TOP_STORIES,
            SectionType.DEEP_DIVE,
            SectionType.INDUSTRY_WATCH,
            SectionType.QUICK_HITS,
        ]

        for section_type in section_order:
            articles = self.context.section_assignments.get(section_type, [])
            if not articles:
                continue

            items = []
            for article in articles:
                summary = self.context.summaries.get(article.id)
                if summary:
                    items.append(summary)

            sections.append(
                NewsletterSection(
                    section_type=section_type,
                    title=section_type.value.replace("_", " ").title(),
                    items=items,
                )
            )

        brand = self.context.brand_config
        edition_id = datetime.now(timezone.utc).strftime("%Y%m%d")

        newsletter = Newsletter(
            edition_id=edition_id,
            subject_line=f"{brand.get('brand', {}).get('name', 'Newsletter')} — {edition_id}",
            sections=sections,
            total_articles=sum(len(s.items) for s in sections),
        )

        self.context.newsletter = newsletter
        self.logger.info(
            "Composed digest newsletter %s with %d articles in %d sections",
            edition_id,
            newsletter.total_articles,
            len(sections),
        )

    async def _compose_single_feature(self) -> None:
        pipeline_cfg = self.context.settings.get("pipeline", {})
        top_k = int(pipeline_cfg.get("single_feature_top_k", 8))
        min_cluster_size = int(pipeline_cfg.get("single_feature_min_cluster_size", 3))

        ranked = sorted(
            [a for a in self.context.articles if a.id in self.context.scores and a.status == ArticleStatus.DEDUPLICATED],
            key=lambda a: self.context.scores[a.id].overall,
            reverse=True,
        )
        candidates = ranked[:top_k]

        signals = [
            {
                "id": a.id,
                "title": _sanitize_untrusted_text(a.title, max_len=240),
                "url": str(a.url),
                "source_name": a.source_name,
                "score": self.context.scores[a.id].overall,
                "summary": _sanitize_untrusted_text((self.context.summaries.get(a.id).summary if a.id in self.context.summaries else a.raw_content), max_len=800),
            }
            for a in candidates
        ]

        stage_1_text = await self._llm_json(
            STAGE_1_CLUSTERING_PROMPT.format(
                signals_json=json.dumps(signals, indent=2),
                min_cluster_size=min_cluster_size,
            ),
            system="You are a precise clustering analyst. Return strict JSON.",
        )

        selected_ids = list(stage_1_text.get("selected_article_ids", []))
        gate = evaluate_single_feature_gate(
            selected_ids=selected_ids,
            all_candidates=candidates,
            min_cluster_size=min_cluster_size,
        )
        if not stage_1_text.get("valid", False):
            gate["ok"] = False
            gate["reasons"].append(stage_1_text.get("reason", "cluster_invalid"))

        if not gate["ok"]:
            reasons = "; ".join(gate["reasons"])
            raise ValueError(f"Single feature policy gate failed: {reasons}")

        selected_articles = [a for a in candidates if a.id in set(selected_ids)]
        cluster_payload = {
            "cluster_label": stage_1_text.get("cluster_label", ""),
            "cluster_rationale": stage_1_text.get("cluster_rationale", ""),
            "selected_signals": [s for s in signals if s["id"] in set(selected_ids)],
        }

        stage_2_text = await self._llm_json(
            STAGE_2_THESIS_PROMPT.format(cluster_json=json.dumps(cluster_payload, indent=2)),
            system="You extract one structural-shift thesis sentence. Return strict JSON.",
        )
        thesis = _require_non_empty_string(stage_2_text.get("thesis", ""), "thesis")

        source_payload = []
        for article in selected_articles:
            source_payload.append(
                {
                    "id": article.id,
                    "title": _sanitize_untrusted_text(article.title, max_len=240),
                    "url": str(article.url),
                    "domain": urlparse(str(article.url)).netloc,
                    "source_name": article.source_name,
                    "content_excerpt": _sanitize_untrusted_text(article.raw_content, max_len=1200),
                    "score": self.context.scores[article.id].overall,
                }
            )

        stage_3_text = await self._llm_json(
            STAGE_3_SYNTHESIS_PROMPT.format(
                thesis=thesis,
                sources_json=json.dumps(source_payload, indent=2),
            ),
            system="You write strategic synthesis in markdown. Return strict JSON.",
        )

        stage_4_text = await self._llm_json(
            STAGE_4_SEO_PROMPT.format(draft_markdown=_require_non_empty_markdown(stage_3_text.get("draft_markdown", ""), "draft_markdown")),
            system="You are an SEO strategist editor. Return strict JSON.",
        )

        edition_id = datetime.now(timezone.utc).strftime("%Y%m%d")
        brand_name = self.context.brand_config.get("brand", {}).get("name", "Newsletter")
        headline = _require_non_empty_string(stage_4_text.get("headline", "Strategic Intelligence Feature"), "headline")
        keyword = _require_non_empty_string(stage_4_text.get("target_keyword", "legal strategy"), "target_keyword")

        newsletter = Newsletter(
            edition_id=edition_id,
            subject_line=f"{brand_name} — {headline}",
            sections=[],
            total_articles=len(selected_articles),
            markdown_body=_require_non_empty_markdown(stage_4_text.get("article_markdown", ""), "article_markdown"),
            metadata={
                "mode": "single_feature",
                "target_keyword": keyword,
                "meta_description": stage_4_text.get("meta_description", ""),
                "thesis": thesis,
                "cluster_label": stage_1_text.get("cluster_label", ""),
                "cluster_article_ids": selected_ids,
                "cluster_rationale": stage_1_text.get("cluster_rationale", ""),
            },
        )
        self.context.newsletter = newsletter
        self.context.curated_articles = selected_articles
        self.logger.info(
            "Composed single-feature newsletter %s with %d source articles",
            edition_id,
            len(selected_articles),
        )

    async def _llm_json(self, prompt: str, system: str) -> dict[str, Any]:
        selected_provider = provider(self.context.settings)

        if selected_provider == "anthropic":
            client = anthropic.AsyncAnthropic()
            model = anthropic_model(self.context.settings)
            response = await client.messages.create(
                model=model,
                max_tokens=2048,
                messages=[{"role": "user", "content": f"{system}\n\n{prompt}"}],
            )
            text = response.content[0].text
        elif selected_provider == "claude_cli":
            model = cli_model(self.context.settings)
            text = await claude_cli_completion(prompt=prompt, system=system, model=model)
        else:
            model = summarizer_model(self.context.settings)
            text = await local_completion(
                model=model,
                prompt=prompt,
                system=system,
                temperature=temperature(self.context.settings),
                max_tokens=2048,
            )

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start : end + 1])
            raise
