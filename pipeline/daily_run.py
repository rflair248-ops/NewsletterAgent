from __future__ import annotations

import asyncio
import logging

from engine.agents.assigner import AssignerAgent
from engine.agents.collector import CollectorAgent
from engine.agents.composer import ComposerAgent
from engine.agents.curator import CuratorAgent
from engine.agents.deduplicator import DeduplicatorAgent
from engine.agents.editor import EditorAgent
from engine.agents.reviewer import ReviewerAgent
from engine.agents.scorer import ScorerAgent
from engine.agents.summarizer import SummarizerAgent
from pipeline.context import PipelineContext
from retrieval.config_loader import load_brand_config, load_settings, load_source_config
from scripts.self_improve import analyze_run

logger = logging.getLogger(__name__)

# Ordered pipeline stages
PIPELINE_AGENTS = [
    CollectorAgent,
    ScorerAgent,
    DeduplicatorAgent,
    CuratorAgent,
    SummarizerAgent,
    AssignerAgent,
    ComposerAgent,
    ReviewerAgent,
    EditorAgent,
]


async def run_pipeline() -> PipelineContext:
    """Execute the full newsletter generation pipeline."""
    settings = load_settings()
    brand = load_brand_config()
    sources = load_source_config()

    context = PipelineContext(
        settings=settings,
        brand_config=brand,
        source_config=sources,
    )

    logger.info("Starting pipeline run %s", context.run_id)

    for agent_cls in PIPELINE_AGENTS:
        agent = agent_cls(context)
        async with agent:
            try:
                await agent.run()
                context.audit(
                    "stage_complete",
                    stage=agent.name,
                    articles_count=len(context.articles),
                )
                if agent.name == "editor":
                    analysis = await analyze_run(context._audit_entries, context.settings)
                    context.audit("self_improvement", suggestions=analysis)
            except Exception:
                logger.exception("Pipeline failed at stage: %s", agent.name)
                context.audit("stage_error", stage=agent.name)
                raise

    context.flush_audit_log()
    logger.info("Pipeline run %s complete", context.run_id)
    return context


def main() -> None:
    """Synchronous entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    asyncio.run(run_pipeline())


if __name__ == "__main__":
    main()
