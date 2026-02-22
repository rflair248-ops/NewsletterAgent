from engine.agents.base import BaseAgent

__all__ = [
    "BaseAgent",
    "CollectorAgent",
    "ScorerAgent",
    "DeduplicatorAgent",
    "CuratorAgent",
    "SummarizerAgent",
    "AssignerAgent",
    "ComposerAgent",
    "ReviewerAgent",
    "EditorAgent",
]


def __getattr__(name: str):  # noqa: ANN001
    """Lazy imports to avoid hard dependency on feedparser/anthropic at import time."""
    _imports = {
        "CollectorAgent": "engine.agents.collector",
        "ScorerAgent": "engine.agents.scorer",
        "DeduplicatorAgent": "engine.agents.deduplicator",
        "CuratorAgent": "engine.agents.curator",
        "SummarizerAgent": "engine.agents.summarizer",
        "AssignerAgent": "engine.agents.assigner",
        "ComposerAgent": "engine.agents.composer",
        "ReviewerAgent": "engine.agents.reviewer",
        "EditorAgent": "engine.agents.editor",
    }
    if name in _imports:
        import importlib

        module = importlib.import_module(_imports[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
