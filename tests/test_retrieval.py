from __future__ import annotations

from retrieval.config_loader import load_brand_config, load_settings, load_source_config


class TestConfigLoader:
    def test_load_settings(self):
        settings = load_settings()
        assert "pipeline" in settings
        assert "llm" in settings
        assert settings["pipeline"]["max_articles_per_run"] == 50

    def test_load_brand_config(self):
        brand = load_brand_config()
        assert "brand" in brand
        assert "voice" in brand
        assert "sections" in brand

    def test_load_source_config(self):
        config = load_source_config()
        assert len(config.rss_feeds) > 0
        assert config.category_weights.get("tech") == 1.0
