from __future__ import annotations

from pathlib import Path

import yaml

from models.source import SourceConfig, SourceFeed

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def load_settings(path: Path | None = None) -> dict:
    """Load pipeline settings from YAML."""
    target = path or CONFIG_DIR / "settings.yaml"
    with target.open() as f:
        return yaml.safe_load(f)


def load_brand_config(path: Path | None = None) -> dict:
    """Load brand voice and editorial config from YAML."""
    target = path or CONFIG_DIR / "brand.yaml"
    with target.open() as f:
        return yaml.safe_load(f)


def load_source_config(path: Path | None = None) -> SourceConfig:
    """Load source registry from YAML and return typed SourceConfig."""
    target = path or CONFIG_DIR / "sources.yaml"
    with target.open() as f:
        data = yaml.safe_load(f)

    sources = data.get("sources", {})
    feeds_raw = sources.get("rss_feeds", [])
    feeds = [SourceFeed(**f) for f in feeds_raw]

    return SourceConfig(
        rss_feeds=feeds,
        category_weights=data.get("category_weights", {}),
    )
