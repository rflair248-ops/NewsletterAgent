from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from models.enums import ArticleStatus, ContentCategory


class Article(BaseModel):
    """A single article flowing through the pipeline."""

    model_config = ConfigDict(frozen=False)

    id: str = Field(description="Deterministic hash of url + published_at")
    url: HttpUrl
    title: str
    source_name: str
    category: ContentCategory = ContentCategory.GENERAL
    published_at: datetime | None = None
    fetched_at: datetime = Field(default_factory=datetime.utcnow)
    raw_content: str = ""
    status: ArticleStatus = ArticleStatus.RAW
    metadata: dict = Field(default_factory=dict)


class ArticleSummary(BaseModel):
    """LLM-generated summary attached to a scored article."""

    article_id: str
    headline: str = Field(max_length=120)
    summary: str = Field(max_length=500)
    key_points: list[str] = Field(default_factory=list)
    read_time_seconds: int = 0
    generated_at: datetime = Field(default_factory=datetime.utcnow)
