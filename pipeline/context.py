from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from memory.mem0_store import Mem0Store
from models.article import Article, ArticleSummary
from models.enums import SectionType
from models.newsletter import Newsletter
from models.score import ArticleScore
from models.source import SourceConfig

logger = logging.getLogger(__name__)


class PipelineContext:
    """Shared state passed through every agent in a single pipeline run."""

    def __init__(
        self,
        settings: dict,
        brand_config: dict,
        source_config: SourceConfig,
        mem0_store: Mem0Store | None = None,
    ) -> None:
        self.settings = settings
        self.brand_config = brand_config
        self.source_config = source_config

        # Memory layer (cross-run dedup, editorial decisions)
        if mem0_store is not None:
            self.memory = mem0_store
        else:
            memory_cfg = dict((settings or {}).get("memory", {}))
            enabled = memory_cfg.pop("enabled", True)
            self.memory = Mem0Store(config=memory_cfg) if enabled else Mem0Store(config={"_disabled": True})

        # Mutable state populated by agents
        self.articles: list[Article] = []
        self.scores: dict[str, ArticleScore] = {}
        self.curated_articles: list[Article] = []
        self.summaries: dict[str, ArticleSummary] = {}
        self.section_assignments: dict[SectionType, list[Article]] = {}
        self.newsletter: Newsletter | None = None
        self.review_result: dict | None = None

        # Audit log buffer
        self._audit_entries: list[dict] = []
        self.run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S")

    def audit(self, event: str, **kwargs: object) -> None:
        """Append a structured audit log entry."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "run_id": self.run_id,
            "event": event,
            **kwargs,
        }
        self._audit_entries.append(entry)
        logger.debug("AUDIT: %s", entry)

    def flush_audit_log(self, path: Path | None = None) -> None:
        """Write buffered audit entries to an NDJSON file."""
        if not self._audit_entries:
            return
        target = path or Path("audit_log") / f"run_{self.run_id}.ndjson"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a") as f:
            for entry in self._audit_entries:
                f.write(json.dumps(entry) + "\n")
        logger.info("Flushed %d audit entries to %s", len(self._audit_entries), target)
        self._audit_entries.clear()
