from __future__ import annotations

import pytest

from models.article import Article
from models.enums import ScoringDimension
from scoring.composite import compute_composite_score
from scoring.quality import score_quality
from scoring.relevance import score_relevance
from scoring.source_authority import score_source_authority
from scoring.timeliness import score_timeliness
from scoring.uniqueness import score_uniqueness


class TestRelevance:
    @pytest.mark.asyncio
    async def test_relevance_scoring(self, sample_article: Article):
        result = await score_relevance(sample_article, {"ai": 1.2, "tech": 1.0})
        assert result.dimension == ScoringDimension.RELEVANCE
        assert 0.0 <= result.value <= 1.0


class TestQuality:
    @pytest.mark.asyncio
    async def test_quality_scoring(self, sample_article: Article):
        result = await score_quality(sample_article)
        assert result.dimension == ScoringDimension.QUALITY
        assert 0.0 <= result.value <= 1.0


class TestTimeliness:
    @pytest.mark.asyncio
    async def test_timeliness_scoring(self, sample_article: Article):
        result = await score_timeliness(sample_article)
        assert result.dimension == ScoringDimension.TIMELINESS
        assert result.value > 0.5  # Recent article should score well


class TestUniqueness:
    @pytest.mark.asyncio
    async def test_uniqueness_scoring(self, sample_article: Article):
        result = await score_uniqueness(sample_article)
        assert result.dimension == ScoringDimension.UNIQUENESS
        assert 0.0 <= result.value <= 1.0


class TestSourceAuthority:
    @pytest.mark.asyncio
    async def test_source_authority(self, sample_article: Article):
        result = await score_source_authority(sample_article)
        assert result.dimension == ScoringDimension.SOURCE_AUTHORITY
        assert result.value == 0.8  # Matches fixture reliability


class TestComposite:
    @pytest.mark.asyncio
    async def test_composite_score(self, sample_article: Article):
        settings = {"pipeline": {"min_quality_score": 0.5}}
        score = await compute_composite_score(
            sample_article,
            category_weights={"ai": 1.2},
            settings=settings,
        )
        assert 0.0 <= score.overall <= 1.0
        assert len(score.breakdown) == 5
