from __future__ import annotations

from pydantic import BaseModel, Field

from models.enums import ScoringDimension


class ScoreBreakdown(BaseModel):
    """Individual dimension score."""

    dimension: ScoringDimension
    value: float = Field(ge=0.0, le=1.0)
    reason: str = ""


class ArticleScore(BaseModel):
    """Composite score for an article."""

    article_id: str
    overall: float = Field(ge=0.0, le=1.0, description="Weighted composite score")
    breakdown: list[ScoreBreakdown] = Field(default_factory=list)
    passed_gate: bool = False

    def dimension_value(self, dim: ScoringDimension) -> float:
        for b in self.breakdown:
            if b.dimension == dim:
                return b.value
        return 0.0
