from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


def validate_brand_compliance(text: str, brand_config: dict) -> list[str]:
    """Check text against brand voice rules. Returns list of violations."""
    violations: list[str] = []
    voice = brand_config.get("voice", {})

    # Check banned phrases
    banned = voice.get("banned_phrases", [])
    text_lower = text.lower()
    for phrase in banned:
        if phrase.lower() in text_lower:
            violations.append(f"Banned phrase detected: '{phrase}'")

    # Check reading level heuristic (sentence length)
    sentences = re.split(r"[.!?]+", text)
    long_sentences = [s for s in sentences if len(s.split()) > 30]
    if long_sentences:
        violations.append(
            f"{len(long_sentences)} sentence(s) exceed 30 words"
        )

    return violations


def validate_newsletter_structure(sections: list, brand_config: dict) -> list[str]:
    """Validate that newsletter sections match brand configuration."""
    violations: list[str] = []
    section_cfg = brand_config.get("sections", [])

    max_items_map = {s["name"]: s.get("max_items", 10) for s in section_cfg}

    for section in sections:
        title = section.title.replace("_", " ").title()
        limit = max_items_map.get(title)
        if limit and len(section.items) > limit:
            violations.append(
                f"Section '{title}' has {len(section.items)} items, max is {limit}"
            )

    return violations
