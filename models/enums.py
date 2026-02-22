from enum import Enum


class ArticleStatus(str, Enum):
    """Tracks an article through the pipeline."""

    RAW = "raw"
    COLLECTED = "collected"
    SCORED = "scored"
    DEDUPLICATED = "deduplicated"
    SUMMARIZED = "summarized"
    ASSIGNED = "assigned"
    EDITED = "edited"
    PUBLISHED = "published"
    REJECTED = "rejected"


class ContentCategory(str, Enum):
    """High-level topic buckets."""

    TECH = "tech"
    AI = "ai"
    RESEARCH = "research"
    BUSINESS = "business"
    GENERAL = "general"


class PipelineStage(str, Enum):
    """Stages in the newsletter generation pipeline."""

    COLLECT = "collect"
    SCORE = "score"
    DEDUPLICATE = "deduplicate"
    CURATE = "curate"
    SUMMARIZE = "summarize"
    ASSIGN = "assign"
    COMPOSE = "compose"
    REVIEW = "review"
    EDIT = "edit"


class ScoringDimension(str, Enum):
    """Dimensions along which articles are scored."""

    RELEVANCE = "relevance"
    QUALITY = "quality"
    TIMELINESS = "timeliness"
    UNIQUENESS = "uniqueness"
    SOURCE_AUTHORITY = "source_authority"


class SectionType(str, Enum):
    """Newsletter section identifiers."""

    TOP_STORIES = "top_stories"
    INDUSTRY_WATCH = "industry_watch"
    DEEP_DIVE = "deep_dive"
    QUICK_HITS = "quick_hits"
