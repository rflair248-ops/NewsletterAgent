from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from engine.llm_router import claude_cli_completion


CRITIC_PROMPT = """\
You are a strict pre-publication QA critic for a single-feature strategic newsletter article.
Evaluate the draft against all required checks and return strict JSON only.

Requirements:
1) H1 is declarative (assertive claim, not a question)
2) Uses 2+ sources
3) Uses 2+ distinct domains
4) Target keyword appears naturally in first 100 words
5) H2 flow covers: What happened -> Why it matters -> What to do
6) Tactical moves count is 3-5
7) Closing is forward-looking (not recap-only)

Return JSON:
{
  "pass": true/false,
  "criteria": {
    "declarative_h1": {"pass": bool, "note": "..."},
    "sources_count": {"pass": bool, "count": int, "note": "..."},
    "domains_count": {"pass": bool, "count": int, "note": "..."},
    "keyword_first_100": {"pass": bool, "note": "..."},
    "h2_flow": {"pass": bool, "note": "..."},
    "tactical_moves_3_to_5": {"pass": bool, "count": int, "note": "..."},
    "forward_looking_close": {"pass": bool, "note": "..."}
  },
  "actionable_fixes": ["specific fix 1", "specific fix 2"]
}

Target keyword: {keyword}
Known source URLs: {source_urls_json}
Draft markdown:
{markdown}
"""

REVISION_PROMPT = """\
You are revising a single-feature strategic newsletter article to satisfy pre-publication QA.

Apply these fixes exactly and keep the article coherent and concise:
{fixes}

Target keyword: {keyword}
Current markdown:
{markdown}

Return the revised markdown only.
"""


def _parse_json_maybe(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _first_100_words(text: str) -> str:
    words = text.split()
    return " ".join(words[:100]).lower()


def _extract_h1(markdown: str) -> str:
    for line in markdown.splitlines():
        if line.strip().startswith("# "):
            return line.strip()[2:].strip()
    return ""


def _extract_h2s(markdown: str) -> list[str]:
    h2s = []
    for line in markdown.splitlines():
        s = line.strip()
        if s.startswith("## "):
            h2s.append(s[3:].strip().lower())
    return h2s


def _count_tactical_moves(markdown: str) -> int:
    lines = [ln.strip() for ln in markdown.splitlines()]
    count = 0
    in_moves = False
    for line in lines:
        if line.startswith("## "):
            in_moves = "tactical" in line.lower() and "move" in line.lower()
            continue
        if in_moves and line[:2].isdigit() and line[1] == ".":
            count += 1
    return count


def rule_based_scorecard(markdown: str, keyword: str, source_urls: list[str]) -> dict[str, Any]:
    h1 = _extract_h1(markdown)
    h1_declarative = bool(h1) and not h1.endswith("?")

    domains = {u.split("//", 1)[-1].split("/", 1)[0].lower() for u in source_urls if "//" in u}
    src_count = len(source_urls)
    domain_count = len(domains)

    first_100 = _first_100_words(markdown)
    keyword_ok = keyword.strip().lower() in first_100 if keyword.strip() else False

    h2s = _extract_h2s(markdown)
    happened = any(any(k in h for k in ["trigger", "what happened", "signal"]) for h in h2s)
    matters = any(any(k in h for k in ["why it matters", "translation", "implication"]) for h in h2s)
    todo = any(any(k in h for k in ["what to do", "tactical", "moves"]) for h in h2s)
    h2_flow_ok = happened and matters and todo

    moves = _count_tactical_moves(markdown)
    moves_ok = 3 <= moves <= 5

    lines = [ln.strip() for ln in markdown.splitlines() if ln.strip()]
    tail = " ".join(lines[-3:]).lower() if lines else ""
    forward_ok = any(k in tail for k in ["next", "ahead", "watch", "position", "future"]) and "in summary" not in tail

    criteria = {
        "declarative_h1": {"pass": h1_declarative, "note": "H1 should be assertive and non-question."},
        "sources_count": {"pass": src_count >= 2, "count": src_count, "note": "Need at least 2 sources."},
        "domains_count": {"pass": domain_count >= 2, "count": domain_count, "note": "Need at least 2 distinct domains."},
        "keyword_first_100": {"pass": keyword_ok, "note": "Keyword should appear naturally in first 100 words."},
        "h2_flow": {"pass": h2_flow_ok, "note": "Need What happened -> Why it matters -> What to do flow."},
        "tactical_moves_3_to_5": {"pass": moves_ok, "count": moves, "note": "Need 3-5 tactical moves."},
        "forward_looking_close": {"pass": forward_ok, "note": "Close should be forward-looking."},
    }

    fixes: list[str] = []
    for key, item in criteria.items():
        if not item.get("pass"):
            fixes.append(f"Fix {key}: {item.get('note', '')}")

    passed = all(c["pass"] for c in criteria.values())
    return {
        "pass": passed,
        "criteria": criteria,
        "actionable_fixes": fixes,
        "critic_mode": "rule_based",
    }


async def critic_scorecard(markdown: str, keyword: str, source_urls: list[str], model: str = "sonnet") -> dict[str, Any]:
    prompt = CRITIC_PROMPT.format(
        keyword=keyword,
        source_urls_json=json.dumps(source_urls),
        markdown=markdown,
    )
    text = await claude_cli_completion(
        prompt=prompt,
        system="Return strict JSON only.",
        model=model,
    )
    score = _parse_json_maybe(text)
    score["critic_mode"] = "claude_cli"
    return score


async def revise_with_fixes(markdown: str, keyword: str, fixes: list[str], model: str = "sonnet") -> str:
    if not fixes:
        return markdown
    prompt = REVISION_PROMPT.format(
        fixes="\n".join(f"- {f}" for f in fixes),
        keyword=keyword,
        markdown=markdown,
    )
    return await claude_cli_completion(
        prompt=prompt,
        system="Return revised markdown only.",
        model=model,
    )


def append_quality_check(scorecard: dict[str, Any], run_id: str, path: Path | None = None) -> Path:
    target = path or Path("output") / "quality_checks.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        **scorecard,
    }
    with target.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return target
