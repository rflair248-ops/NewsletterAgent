from __future__ import annotations

from typing import Any

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
DEFAULT_CLI_MODEL = "sonnet"
DEFAULT_LOCAL_SUMMARIZER_MODEL = "mistral-small"
DEFAULT_LOCAL_EDITOR_MODEL = "llama3.1:8b"
DEFAULT_LOCAL_MEM0_LLM_MODEL = DEFAULT_LOCAL_EDITOR_MODEL
DEFAULT_LOCAL_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.3
DEFAULT_SCORING_WEIGHTS = {
    "relevance": 0.25,
    "quality": 0.25,
    "timeliness": 0.20,
    "uniqueness": 0.15,
    "source_authority": 0.15,
}


def llm_cfg(settings: dict[str, Any]) -> dict[str, Any]:
    return settings.get("llm", {}) if isinstance(settings, dict) else {}


def local_cfg(settings: dict[str, Any]) -> dict[str, Any]:
    cfg = llm_cfg(settings).get("local", {})
    return cfg if isinstance(cfg, dict) else {}


def provider(settings: dict[str, Any]) -> str:
    return str(local_cfg(settings).get("provider", "ollama"))


def anthropic_model(settings: dict[str, Any]) -> str:
    return str(llm_cfg(settings).get("model") or DEFAULT_ANTHROPIC_MODEL)


def cli_model(settings: dict[str, Any]) -> str:
    return str(llm_cfg(settings).get("cli_model") or DEFAULT_CLI_MODEL)


def summarizer_model(settings: dict[str, Any]) -> str:
    return str(local_cfg(settings).get("summarizer_model") or DEFAULT_LOCAL_SUMMARIZER_MODEL)


def editor_model(settings: dict[str, Any]) -> str:
    return str(local_cfg(settings).get("editor_model") or DEFAULT_LOCAL_EDITOR_MODEL)


def mem0_llm_model(settings: dict[str, Any]) -> str:
    return str(local_cfg(settings).get("mem0_llm_model") or DEFAULT_LOCAL_MEM0_LLM_MODEL)


def scoring_weights(settings: dict[str, Any]) -> dict[str, float]:
    configured = settings.get("scoring", {}).get("dimension_weights", {}) if isinstance(settings, dict) else {}
    if not isinstance(configured, dict):
        configured = {}

    merged = dict(DEFAULT_SCORING_WEIGHTS)
    for key, default_value in DEFAULT_SCORING_WEIGHTS.items():
        value = configured.get(key, default_value)
        try:
            merged[key] = float(value)
        except (TypeError, ValueError):
            merged[key] = default_value
    return merged


def local_max_tokens(settings: dict[str, Any], default: int = DEFAULT_LOCAL_MAX_TOKENS) -> int:
    return int(local_cfg(settings).get("max_tokens", default))


def temperature(settings: dict[str, Any], default: float = DEFAULT_TEMPERATURE) -> float:
    return float(local_cfg(settings).get("temperature", default))
