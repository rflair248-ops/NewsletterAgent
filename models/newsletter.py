from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from models.article import ArticleSummary
from models.enums import SectionType


class NewsletterSection(BaseModel):
    """One section of the final newsletter."""

    section_type: SectionType
    title: str
    items: list[ArticleSummary] = Field(default_factory=list)
    intro_text: str = ""


class Newsletter(BaseModel):
    """The final assembled newsletter."""

    edition_id: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    subject_line: str = ""
    sections: list[NewsletterSection] = Field(default_factory=list)
    html_body: str = ""
    markdown_body: str = ""
    total_articles: int = 0
    metadata: dict = Field(default_factory=dict)
