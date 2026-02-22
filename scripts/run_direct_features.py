from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import httpx
import yaml

from engine.llm_router import claude_cli_completion
from policy.quality_checks import critic_scorecard


async def fetch_feeds() -> list[dict]:
    src = yaml.safe_load(Path("config/sources.yaml").read_text())
    out: list[dict] = []
    async with httpx.AsyncClient(timeout=30, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0 (NewsletterAgent/1.0)"}) as client:
        for f in src.get("sources", {}).get("rss_feeds", []):
            try:
                r = await client.get(f["url"])
                r.raise_for_status()
                parsed = feedparser.parse(r.text)
                for e in parsed.entries[:20]:
                    out.append(
                        {
                            "title": e.get("title", ""),
                            "url": e.get("link", ""),
                            "summary": (e.get("summary", "") or "")[:600],
                            "source": f["name"],
                            "category": f.get("category", "general"),
                        }
                    )
            except Exception:
                continue
    return [x for x in out if x.get("url")][:120]


async def generate_article(candidates: list[dict]) -> tuple[str, str, list[str], str]:
    prompt = f"""Write ONE Search Engine Land-style SEO strategic article for law firm operators.
Use today's signals below and choose a strong topic with at least 2 domains and 2 independent sources.

Output format exactly:
PRIMARY_KEYWORD: <keyword>
TITLE: <title>
<markdown article>

Requirements:
- Declarative H1
- First 100 words include PRIMARY_KEYWORD naturally
- H2 flow: What happened / Why it matters / What to do
- Include a numbered tactical section with 3-5 moves
- Forward-looking close
- End with:
## Sources
- <url>
- <url>

Signals JSON:
{str(candidates)[:45000]}
"""
    text = await claude_cli_completion(prompt=prompt, system="You are an expert SEO editor.", model="sonnet")
    m_kw = re.search(r"PRIMARY_KEYWORD:\s*(.+)", text)
    m_title = re.search(r"TITLE:\s*(.+)", text)
    keyword = m_kw.group(1).strip() if m_kw else "law firm seo"
    title = m_title.group(1).strip() if m_title else "Strategic SEO Shift for Law Firms"
    md = re.sub(r"^PRIMARY_KEYWORD:.*\n?", "", text)
    md = re.sub(r"^TITLE:.*\n?", "", md)
    urls = re.findall(r"https?://[^\s)]+", md)
    return title, keyword, list(dict.fromkeys(urls)), md.strip()


async def run() -> None:
    items = await fetch_feeds()
    out_dir = Path("output/direct_features")
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(out_dir.glob("feature_*.md"))
    produced = len(existing)
    attempts = 0
    while produced < 2 and attempts < 20:
        attempts += 1
        title, kw, urls, md = await generate_article(items)
        score = await critic_scorecard(markdown=md, keyword=kw, source_urls=urls, model="sonnet")
        if not score.get("pass"):
            continue
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        fp = out_dir / f"feature_{produced+1}_{ts}.md"
        fp.write_text(md, encoding="utf-8")
        produced += 1
        print(f"WROTE {fp}")
        print(f"TITLE: {title}")

    if produced < 2:
        raise RuntimeError("Could not produce 2 quality-passing articles in attempt budget")


if __name__ == "__main__":
    asyncio.run(run())
