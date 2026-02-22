from __future__ import annotations

from datetime import datetime

from models.article import Article, ArticleSummary
from models.enums import ArticleStatus, ContentCategory, ScoringDimension, SectionType
from models.newsletter import Newsletter, NewsletterSection
from models.score import ArticleScore, ScoreBreakdown
from models.source import SourceConfig, SourceFeed


class TestArticle:
    def test_create_article(self):
        article = Article(
            id="abc123",
            url="https://example.com/test",
            title="Test Article",
            source_name="Test",
        )
        assert article.id == "abc123"
        assert article.status == ArticleStatus.RAW
        assert article.category == ContentCategory.GENERAL

    def test_article_summary(self):
        summary = ArticleSummary(
            article_id="abc123",
            headline="A Short Headline",
            summary="Article summary text.",
            key_points=["Point 1", "Point 2"],
        )
        assert summary.article_id == "abc123"
        assert len(summary.key_points) == 2


class TestScore:
    def test_score_breakdown(self):
        bd = ScoreBreakdown(
            dimension=ScoringDimension.RELEVANCE,
            value=0.85,
            reason="High relevance",
        )
        assert bd.value == 0.85

    def test_article_score_dimension_lookup(self):
        score = ArticleScore(
            article_id="test1",
            overall=0.7,
            breakdown=[
                ScoreBreakdown(
                    dimension=ScoringDimension.RELEVANCE, value=0.8, reason=""
                ),
                ScoreBreakdown(
                    dimension=ScoringDimension.QUALITY, value=0.6, reason=""
                ),
            ],
        )
        assert score.dimension_value(ScoringDimension.RELEVANCE) == 0.8
        assert score.dimension_value(ScoringDimension.TIMELINESS) == 0.0


class TestNewsletter:
    def test_create_newsletter(self):
        nl = Newsletter(edition_id="20260222")
        assert nl.total_articles == 0
        assert nl.sections == []

    def test_newsletter_section(self):
        section = NewsletterSection(
            section_type=SectionType.TOP_STORIES,
            title="Top Stories",
        )
        assert section.section_type == SectionType.TOP_STORIES


class TestSource:
    def test_source_feed(self):
        feed = SourceFeed(
            name="Test",
            url="https://example.com/feed",
            category="tech",
            reliability=0.85,
        )
        assert feed.reliability == 0.85

    def test_source_config(self):
        config = SourceConfig(
            rss_feeds=[],
            category_weights={"tech": 1.0},
        )
        assert config.category_weights["tech"] == 1.0
