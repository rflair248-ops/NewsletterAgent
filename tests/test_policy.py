from __future__ import annotations

from models.article import Article
from models.enums import ArticleStatus, ContentCategory, ScoringDimension
from models.score import ArticleScore, ScoreBreakdown
from policy.gates import passes_publication_gate
from policy.validators import validate_brand_compliance


class TestPublicationGate:
    def test_passes_with_good_scores(self):
        article = Article(
            id="test1",
            url="https://example.com/1",
            title="Good Article",
            source_name="Test",
        )
        score = ArticleScore(
            article_id="test1",
            overall=0.8,
            breakdown=[
                ScoreBreakdown(dimension=ScoringDimension.RELEVANCE, value=0.8, reason=""),
                ScoreBreakdown(dimension=ScoringDimension.QUALITY, value=0.7, reason=""),
            ],
        )
        assert passes_publication_gate(article, score) is True

    def test_rejects_low_relevance(self):
        article = Article(
            id="test2",
            url="https://example.com/2",
            title="Low Relevance",
            source_name="Test",
        )
        score = ArticleScore(
            article_id="test2",
            overall=0.3,
            breakdown=[
                ScoreBreakdown(dimension=ScoringDimension.RELEVANCE, value=0.2, reason=""),
                ScoreBreakdown(dimension=ScoringDimension.QUALITY, value=0.7, reason=""),
            ],
        )
        assert passes_publication_gate(article, score) is False


class TestBrandCompliance:
    def test_detects_banned_phrases(self):
        text = "This game-changer is truly revolutionary"
        brand = {"voice": {"banned_phrases": ["game-changer", "revolutionary"]}}
        violations = validate_brand_compliance(text, brand)
        assert len(violations) == 2

    def test_passes_clean_text(self):
        text = "A well-written article about technology."
        brand = {"voice": {"banned_phrases": ["game-changer"]}}
        violations = validate_brand_compliance(text, brand)
        assert len(violations) == 0
