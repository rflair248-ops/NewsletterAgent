"""Post-run self-improvement: analyze what worked and tune scoring weights."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import anthropic

from engine.llm_router import local_completion
from retrieval.config_loader import load_settings

logger = logging.getLogger(__name__)

IMPROVEMENT_LOG = Path("output/improvements.jsonl")
AUDIT_GLOB = Path("audit_log")


def _load_recent_audit_entries(limit: int = 50) -> list[dict[str, Any]]:
    files = sorted(AUDIT_GLOB.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not files:
        return []
    latest = files[-1]
    entries: list[dict[str, Any]] = []
    with latest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries[-limit:]


async def analyze_run(audit_entries: list[dict[str, Any]], settings: dict[str, Any]) -> dict[str, Any]:
    """Ask local model to analyze pipeline performance and suggest tuning."""
    summary = json.dumps(audit_entries[-50:], indent=2, default=str)
    prompt = f"""Analyze this newsletter pipeline run audit log and suggest improvements.

Audit log (last 50 entries):
{summary}

Current scoring weights:
relevance: 0.25, quality: 0.25, timeliness: 0.20, uniqueness: 0.15, source_authority: 0.15

Respond with JSON:
{{
  "observations": ["..."],
  "suggested_weight_changes": {{"dimension": 0.00}},
  "suggested_threshold_changes": {{"param": 0.00}},
  "reasoning": "..."
}}"""

    llm_cfg = settings.get("llm", {})
    local_cfg = llm_cfg.get("local", {})
    provider = local_cfg.get("provider", "ollama")

    if provider == "anthropic":
        client = anthropic.AsyncAnthropic()
        model = llm_cfg.get("model", "claude-sonnet-4-20250514")
        response = await client.messages.create(
            model=model,
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}],
        )
        result = response.content[0].text
    else:
        model = local_cfg.get("editor_model", "llama3.1:8b")
        result = await local_completion(
            model=model,
            prompt=prompt,
            system="You are a newsletter quality analyst. Be concise and data-driven.",
        )

    try:
        analysis = json.loads(result)
    except json.JSONDecodeError:
        analysis = {"raw_response": result, "parse_error": True}

    IMPROVEMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(IMPROVEMENT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(analysis, default=str) + "\n")

    logger.info("Self-improvement analysis complete: %d observations", len(analysis.get("observations", [])))
    return analysis


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    entries = _load_recent_audit_entries(limit=50)
    if not entries:
        logger.warning("No audit entries found. Nothing to improve.")
        return

    settings = load_settings()
    await analyze_run(entries, settings)


if __name__ == "__main__":
    asyncio.run(main())
