from __future__ import annotations

from pipeline.context import PipelineContext
from models.source import SourceConfig
from retrieval.config_loader import load_settings, load_brand_config


class TestPipelineContext:
    def test_context_creation(self):
        ctx = PipelineContext(
            settings=load_settings(),
            brand_config=load_brand_config(),
            source_config=SourceConfig(),
        )
        assert ctx.articles == []
        assert ctx.scores == {}
        assert ctx.newsletter is None
        assert ctx.run_id is not None

    def test_audit_logging(self):
        ctx = PipelineContext(
            settings=load_settings(),
            brand_config=load_brand_config(),
            source_config=SourceConfig(),
        )
        ctx.audit("test_event", article_id="abc", reason="testing")
        assert len(ctx._audit_entries) == 1
        assert ctx._audit_entries[0]["event"] == "test_event"
