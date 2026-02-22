from __future__ import annotations

from typing import Any

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
DEFAULT_CLI_MODEL = "sonnet"
DEFAULT_LOCAL_SUMMARIZER_MODEL = "mistral-small"
DEFAULT_LOCAL_EDITOR_MODEL = "llama3.1:8b"
DEFAULT_LOCAL_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.3


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


def local_max_tokens(settings: dict[str, Any], default: int = DEFAULT_LOCAL_MAX_TOKENS) -> int:
    return int(local_cfg(settings).get("max_tokens", default))


def temperature(settings: dict[str, Any], default: float = DEFAULT_TEMPERATURE) -> float:
    return float(local_cfg(settings).get("temperature", default))
