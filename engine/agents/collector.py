from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import feedparser
import httpx

from engine.agents.base import BaseAgent
from models.article import Article
from models.enums import ArticleStatus, ContentCategory

if TYPE_CHECKING:
    from models.source import SourceFeed


class CollectorAgent(BaseAgent):
    """Fetches articles from configured RSS feeds and APIs."""

    name = "collector"

    async def run(self) -> None:
        sources = self.context.source_config
        articles: list[Article] = []

        for feed in sources.rss_feeds:
            try:
                fetched = await self._fetch_feed(feed)
                articles.extend(fetched)
                self.logger.info("Collected %d articles from %s", len(fetched), feed.name)
            except Exception:
                self.logger.exception("Failed to fetch feed %s", feed.name)

        self.context.articles = articles
        self.logger.info("Total articles collected: %d", len(articles))

    async def _fetch_feed(self, feed: SourceFeed) -> list[Article]:
        headers = {"User-Agent": "Mozilla/5.0 (NewsletterAgent/1.0)"}
        async with httpx.AsyncClient(timeout=30, follow_redirects=True, headers=headers) as client:
            resp = await client.get(str(feed.url))
            resp.raise_for_status()

        parsed = feedparser.parse(resp.text)
        articles: list[Article] = []

        for entry in parsed.entries:
            url = entry.get("link", "")
            if not url:
                continue

            published = entry.get("published_parsed")
            pub_dt = datetime(*published[:6], tzinfo=timezone.utc) if published else None

            article_id = hashlib.sha256(
                f"{url}:{pub_dt or ''}".encode()
            ).hexdigest()[:16]

            category = ContentCategory(feed.category) if feed.category in ContentCategory.__members__.values() else ContentCategory.GENERAL

            articles.append(
                Article(
                    id=article_id,
                    url=url,
                    title=entry.get("title", "Untitled"),
                    source_name=feed.name,
                    category=category,
                    published_at=pub_dt,
                    raw_content=entry.get("summary", ""),
                    status=ArticleStatus.COLLECTED,
                    metadata={"feed_url": str(feed.url), "reliability": feed.reliability},
                )
            )

        return articles[: self.context.settings.get("pipeline", {}).get("max_articles_per_run", 50)]
