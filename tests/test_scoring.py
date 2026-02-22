from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from memory.mem0_store import Mem0Store
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
    async def test_uniqueness_scoring_without_memory(self, sample_article: Article):
        result = await score_uniqueness(sample_article)
        assert result.dimension == ScoringDimension.UNIQUENESS
        assert 0.0 <= result.value <= 1.0

    @pytest.mark.asyncio
    async def test_uniqueness_penalized_by_mem0_match(self, sample_article: Article):
        mock_store = MagicMock(spec=Mem0Store)
        mock_store.enabled = True
        mock_store.find_similar_articles = AsyncMock(
            return_value=[{"memory_id": "m1", "score": 0.92, "memory": "seen", "metadata": {}}]
        )

        result_with_mem0 = await score_uniqueness(sample_article, mem0_store=mock_store)
        result_without = await score_uniqueness(sample_article, mem0_store=None)

        assert result_with_mem0.value < result_without.value
        assert "Mem0 match" in result_with_mem0.reason

    @pytest.mark.asyncio
    async def test_uniqueness_partial_mem0_match(self, sample_article: Article):
        mock_store = MagicMock(spec=Mem0Store)
        mock_store.enabled = True
        mock_store.find_similar_articles = AsyncMock(
            return_value=[{"memory_id": "m1", "score": 0.70, "memory": "partial", "metadata": {}}]
        )

        result = await score_uniqueness(sample_article, mem0_store=mock_store)
        assert "Mem0 partial match" in result.reason


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

    @pytest.mark.asyncio
    async def test_composite_score_with_mem0(self, sample_article: Article):
        mock_store = MagicMock(spec=Mem0Store)
        mock_store.enabled = True
        mock_store.find_similar_articles = AsyncMock(return_value=[])

        settings = {"pipeline": {"min_quality_score": 0.5}}
        score = await compute_composite_score(
            sample_article,
            category_weights={"ai": 1.2},
            settings=settings,
            mem0_store=mock_store,
        )
        assert 0.0 <= score.overall <= 1.0
        assert len(score.breakdown) == 5
