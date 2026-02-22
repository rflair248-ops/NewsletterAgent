from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class Mem0Store:
    """Cross-run memory layer backed by Mem0.

    Provides three capabilities:
    1. Article memory — remember articles seen across runs for deduplication
    2. Decision memory — remember editorial decisions for consistency
    3. Similarity search — find previously seen articles similar to a candidate

    Uses the open-source ``mem0ai.Memory`` class with local Qdrant storage
    by default. Set ``MEM0_API_KEY`` to use the managed platform instead.
    """

    def __init__(
        self,
        api_key: str | None = None,
        config: dict | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("MEM0_API_KEY")
        self._client: Any = None
        self._enabled = False
        self._init_client(config)

    def _init_client(self, config: dict | None) -> None:
        """Lazily initialize the Mem0 client."""
        try:
            if self.api_key:
                from mem0 import MemoryClient

                self._client = MemoryClient(api_key=self.api_key)
                self._enabled = True
                logger.info("Mem0 managed platform client initialized")
            else:
                from mem0 import Memory

                mem_config = config or {
                    "vector_store": {
                        "provider": "qdrant",
                        "config": {

                            "path": ".mem0/qdrant",
                            "on_disk": True,
                        },
                    },
                    "embedder": {
                        "provider": "ollama",
                        "config": {
                            "model": "nomic-embed-text",
                            "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                        },
                    },
                    "llm": {
                        "provider": "ollama",
                        "config": {
                            "model": "llama3.1:8b",
                            "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                        },
                    },
                }
                self._client = Memory.from_config(mem_config)
                self._enabled = True
                logger.info("Mem0 open-source client initialized (local Qdrant)")
        except ImportError:
            logger.warning(
                "mem0ai package not installed — memory features disabled. "
                "Install with: pip install 'newsletter-agent[memory]'"
            )
            self._enabled = False
        except Exception:
            logger.exception("Failed to initialize Mem0 client")
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def store_article(
        self,
        article_id: str,
        title: str,
        source: str,
        summary: str = "",
        run_id: str = "",
    ) -> str | None:
        """Store an article in memory for future cross-run dedup.

        Returns the mem0 memory ID if successful, None otherwise.
        """
        if not self._enabled:
            return None

        messages = [
            {
                "role": "user",
                "content": (
                    f"Article '{title}' from {source}. "
                    f"Summary: {summary[:500]}"
                ),
            }
        ]

        try:
            result = self._client.add(
                messages,
                user_id="newsletter_pipeline",
                metadata={
                    "article_id": article_id,
                    "source": source,
                    "run_id": run_id,
                    "type": "article",
                },
            )
            mem_id = _extract_memory_id(result)
            logger.debug("Stored article %s in Mem0 (memory_id=%s)", article_id, mem_id)
            return mem_id
        except Exception:
            logger.exception("Failed to store article %s in Mem0", article_id)
            return None

    async def find_similar_articles(
        self,
        title: str,
        content: str = "",
        limit: int = 5,
    ) -> list[dict]:
        """Search memory for previously seen articles similar to the query.

        Returns a list of dicts with keys: article_id, score, memory.
        """
        if not self._enabled:
            return []

        query = f"{title}. {content[:300]}" if content else title

        try:
            results = self._client.search(
                query,
                filters={"user_id": "newsletter_pipeline"},
                limit=limit,
            )
            return _normalize_search_results(results)
        except Exception:
            logger.exception("Mem0 similarity search failed")
            return []

    async def store_decision(
        self,
        run_id: str,
        article_id: str,
        decision: str,
        reason: str = "",
    ) -> str | None:
        """Store an editorial decision for future reference.

        Helps maintain consistency across runs (e.g. "we always cover
        articles from this source" or "we rejected this topic before").
        """
        if not self._enabled:
            return None

        messages = [
            {
                "role": "user",
                "content": (
                    f"Editorial decision for article {article_id}: {decision}. "
                    f"Reason: {reason}"
                ),
            }
        ]

        try:
            result = self._client.add(
                messages,
                user_id="newsletter_editorial",
                metadata={
                    "article_id": article_id,
                    "decision": decision,
                    "run_id": run_id,
                    "type": "decision",
                },
            )
            mem_id = _extract_memory_id(result)
            logger.debug("Stored decision for %s: %s (memory_id=%s)", article_id, decision, mem_id)
            return mem_id
        except Exception:
            logger.exception("Failed to store decision for %s", article_id)
            return None

    async def get_past_decisions(
        self,
        query: str = "editorial decisions",
        limit: int = 10,
    ) -> list[dict]:
        """Retrieve past editorial decisions relevant to a query."""
        if not self._enabled:
            return []

        try:
            results = self._client.search(
                query,
                filters={"user_id": "newsletter_editorial"},
                limit=limit,
            )
            return _normalize_search_results(results)
        except Exception:
            logger.exception("Failed to retrieve past decisions")
            return []

    async def get_all_memories(self, user_id: str = "newsletter_pipeline") -> list[dict]:
        """Retrieve all stored memories for a given user scope."""
        if not self._enabled:
            return []

        try:
            result = self._client.get_all(user_id=user_id)
            if isinstance(result, dict):
                return result.get("results", result.get("memories", []))
            if isinstance(result, list):
                return result
            return []
        except Exception:
            logger.exception("Failed to retrieve all memories")
            return []


def _extract_memory_id(result: Any) -> str | None:
    """Extract the memory ID from a mem0 add() response."""
    if isinstance(result, dict):
        results = result.get("results", [])
        if results and isinstance(results, list):
            return results[0].get("id")
        return result.get("id")
    if isinstance(result, list) and result:
        return result[0].get("id")
    return None


def _normalize_search_results(results: Any) -> list[dict]:
    """Normalize mem0 search results into a consistent format."""
    normalized = []

    items = results
    if isinstance(results, dict):
        items = results.get("results", results.get("memories", []))

    for item in items:
        if isinstance(item, dict):
            normalized.append({
                "memory_id": item.get("id", ""),
                "memory": item.get("memory", item.get("text", "")),
                "score": item.get("score", item.get("relevance", 0.0)),
                "metadata": item.get("metadata", {}),
            })

    return normalized
