from __future__ import annotations

from pydantic import BaseModel, Field, HttpUrl


class SourceFeed(BaseModel):
    """Configuration for a single RSS/Atom feed."""

    name: str
    url: HttpUrl
    category: str = "general"
    reliability: float = Field(ge=0.0, le=1.0, default=0.7)
    refresh_minutes: int = 60


class SourceConfig(BaseModel):
    """Top-level source registry loaded from sources.yaml."""

    rss_feeds: list[SourceFeed] = Field(default_factory=list)
    category_weights: dict[str, float] = Field(default_factory=dict)
