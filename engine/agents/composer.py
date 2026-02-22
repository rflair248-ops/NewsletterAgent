from __future__ import annotations

from datetime import datetime, timezone

from engine.agents.base import BaseAgent
from models.enums import SectionType
from models.newsletter import Newsletter, NewsletterSection


class ComposerAgent(BaseAgent):
    """Assembles section assignments and summaries into a Newsletter object."""

    name = "composer"

    async def run(self) -> None:
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
            "Composed newsletter %s with %d articles in %d sections",
            edition_id,
            newsletter.total_articles,
            len(sections),
        )
