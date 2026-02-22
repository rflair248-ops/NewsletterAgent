from __future__ import annotations

from engine.agents.base import BaseAgent
from models.enums import ArticleStatus, SectionType


class AssignerAgent(BaseAgent):
    """Assigns curated articles to newsletter sections based on scores and config."""

    name = "assigner"

    async def run(self) -> None:
        brand = self.context.brand_config
        sections_cfg = brand.get("sections", [])

        section_limits = {}
        for sec in sections_cfg:
            section_type = self._map_section(sec["name"])
            if section_type:
                section_limits[section_type] = sec.get("max_items", 3)

        assignments: dict[SectionType, list] = {st: [] for st in SectionType}

        # Sort articles by score descending
        ranked = sorted(
            self.context.curated_articles,
            key=lambda a: self.context.scores.get(a.id, None) and self.context.scores[a.id].overall or 0,
            reverse=True,
        )

        for article in ranked:
            assigned = False
            # Top stories get the highest-scored articles
            if (
                len(assignments[SectionType.TOP_STORIES])
                < section_limits.get(SectionType.TOP_STORIES, 3)
            ):
                assignments[SectionType.TOP_STORIES].append(article)
                assigned = True
            # Deep dive: next best unassigned
            elif (
                len(assignments[SectionType.DEEP_DIVE])
                < section_limits.get(SectionType.DEEP_DIVE, 1)
            ):
                assignments[SectionType.DEEP_DIVE].append(article)
                assigned = True
            # Industry watch
            elif (
                len(assignments[SectionType.INDUSTRY_WATCH])
                < section_limits.get(SectionType.INDUSTRY_WATCH, 4)
            ):
                assignments[SectionType.INDUSTRY_WATCH].append(article)
                assigned = True
            # Quick hits: everything else
            elif (
                len(assignments[SectionType.QUICK_HITS])
                < section_limits.get(SectionType.QUICK_HITS, 4)
            ):
                assignments[SectionType.QUICK_HITS].append(article)
                assigned = True

            if assigned:
                article.status = ArticleStatus.ASSIGNED

        self.context.section_assignments = assignments
        total = sum(len(v) for v in assignments.values())
        self.logger.info("Assigned %d articles across %d sections", total, len(assignments))

    @staticmethod
    def _map_section(name: str) -> SectionType | None:
        mapping = {
            "Top Stories": SectionType.TOP_STORIES,
            "Industry Watch": SectionType.INDUSTRY_WATCH,
            "Deep Dive": SectionType.DEEP_DIVE,
            "Quick Hits": SectionType.QUICK_HITS,
        }
        return mapping.get(name)
