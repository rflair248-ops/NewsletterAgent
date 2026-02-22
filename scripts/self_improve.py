"""Post-run self-improvement: analyze what worked and tune scoring weights."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from engine.llm_router import local_completion

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


def _extract_json_block(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    return {
        "observations": ["Could not parse model response as strict JSON."],
        "suggested_weight_changes": {},
        "suggested_threshold_changes": {},
        "reasoning": text[:2000],
    }


async def analyze_run(audit_entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Ask local model to analyze pipeline performance and suggest tuning."""
    summary = json.dumps(audit_entries[-50:], indent=2, default=str)
    model = "mistral-small"

    prompt = f"""Analyze this newsletter pipeline run audit log and suggest improvements.

Audit log (last 50 entries):
{summary}

Current scoring weights:
- relevance: 0.25
- quality: 0.25
- timeliness: 0.20
- uniqueness: 0.15
- source_authority: 0.15

Respond with strict JSON:
{{
  "observations": ["..."],
  "suggested_weight_changes": {{"dimension": 0.00}},
  "suggested_threshold_changes": {{"param": 0.00}},
  "reasoning": "..."
}}
"""

    raw = await local_completion(model=model, prompt=prompt, temperature=0.2, max_tokens=1200)
    result = _extract_json_block(raw)
    return result


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    entries = _load_recent_audit_entries(limit=50)
    if not entries:
        logger.warning("No audit entries found. Nothing to improve.")
        return

    analysis = await analyze_run(entries)

    IMPROVEMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "audit_entries_analyzed": len(entries),
        "analysis": analysis,
    }
    with IMPROVEMENT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info("Wrote self-improvement analysis to %s", IMPROVEMENT_LOG)


if __name__ == "__main__":
    asyncio.run(main())
