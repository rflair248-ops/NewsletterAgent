from __future__ import annotations

from datetime import datetime, timezone

import pytest

from models.article import Article
from models.enums import ArticleStatus, ContentCategory
from models.source import SourceConfig, SourceFeed
from pipeline.context import PipelineContext
from retrieval.config_loader import load_brand_config, load_settings


@pytest.fixture
def sample_article() -> Article:
    return Article(
        id="test123",
        url="https://example.com/article-1",
        title="Test Article About AI Advances",
        source_name="Test Source",
        category=ContentCategory.AI,
        published_at=datetime.now(timezone.utc),
        raw_content="This is a test article about recent advances in artificial intelligence. "
        "Researchers have developed new methods for training large language models that "
        "significantly reduce computational costs while maintaining performance.",
        status=ArticleStatus.COLLECTED,
        metadata={"reliability": 0.8, "feed_url": "https://example.com/feed"},
    )


@pytest.fixture
def sample_articles() -> list[Article]:
    return [
        Article(
            id=f"art{i}",
            url=f"https://example.com/article-{i}",
            title=f"Test Article {i}",
            source_name="Test Source",
            category=ContentCategory.TECH,
            published_at=datetime.now(timezone.utc),
            raw_content=f"Content for article {i}. " * 20,
            status=ArticleStatus.COLLECTED,
            metadata={"reliability": 0.7},
        )
        for i in range(5)
    ]


@pytest.fixture
def source_config() -> SourceConfig:
    return SourceConfig(
        rss_feeds=[
            SourceFeed(
                name="Test Feed",
                url="https://example.com/feed",
                category="tech",
                reliability=0.8,
            ),
        ],
        category_weights={"tech": 1.0, "ai": 1.2, "general": 0.8},
    )


@pytest.fixture
def pipeline_context(source_config: SourceConfig) -> PipelineContext:
    settings = load_settings()
    brand = load_brand_config()
    return PipelineContext(
        settings=settings,
        brand_config=brand,
        source_config=source_config,
    )
