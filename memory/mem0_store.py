from __future__ import annotations

import asyncio
import logging
from functools import partial
import os
import shutil
from pathlib import Path
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

    PIPELINE_USER_ID = "newsletter_pipeline"
    EDITORIAL_USER_ID = "newsletter_editorial"
    AGENT_ID = "newsletter_agent"

    def __init__(
        self,
        api_key: str | None = None,
        config: dict | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("MEM0_API_KEY")
        self._client: Any = None
        self._enabled = False
        self._warned_errors: set[str] = set()
        self._init_client(config)

    def _init_client(self, config: dict | None) -> None:
        """Lazily initialize the Mem0 client."""
        if config and config.get("_disabled"):
            logger.info("Mem0 disabled by config")
            self._enabled = False
            return

        try:
            if self.api_key:
                from mem0 import MemoryClient

                self._client = MemoryClient(api_key=self.api_key)
                self._enabled = True
                logger.info("Mem0 managed platform client initialized")
            else:
                from mem0 import Memory

                mem_config = self._build_local_config(config)
                self._maybe_reset_local_store(mem_config)
                self._client = Memory.from_config(mem_config)
                self._enabled = True
                logger.info("Mem0 open-source client initialized (local Qdrant)")
        except ImportError:
            logger.warning(
                "mem0ai package not installed — memory features disabled. "
                "Install with: pip install 'newsletter-agent[memory]'"
            )
            self._enabled = False
        except Exception as exc:
            self._warn_once("init", f"Failed to initialize Mem0 client: {exc}")
            logger.debug("Mem0 init failure details", exc_info=True)
            self._enabled = False

    def _build_local_config(self, config: dict | None) -> dict:
        """Build local Mem0 config with stable embedding dimensions and collection isolation."""
        model = os.getenv("MEM0_EMBED_MODEL", "nomic-embed-text")
        embedding_dims = int(os.getenv("MEM0_EMBED_DIMS", "768"))
        vector_path = os.getenv("MEM0_VECTOR_PATH", f".mem0/qdrant_{embedding_dims}d_v2")

        # Use a v2 collection default so old 1536-dim collections cannot collide.
        default_collection = f"newsletter_agent_{embedding_dims}d_v2"
        collection_name = os.getenv("MEM0_COLLECTION_NAME", default_collection)

        base = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "path": vector_path,
                    "collection_name": collection_name,
                    "on_disk": True,
                },
            },
            "embedder": {
                "provider": "ollama",
                "config": {
                    "model": model,
                    "embedding_dims": embedding_dims,
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

        if config:
            normalized = dict(config)
            vs = dict(normalized.get("vector_store", {}))
            if "path" in vs or "collection_name" in vs:
                vs_cfg = dict(vs.get("config", {}))
                if "path" in vs:
                    path_value = vs.pop("path")
                    # Migrate legacy default path to a dimension-scoped store.
                    if str(path_value).rstrip("/") == ".mem0/qdrant":
                        path_value = f".mem0/qdrant_{embedding_dims}d_v2"
                    vs_cfg["path"] = path_value
                if "collection_name" in vs:
                    vs_cfg["collection_name"] = vs.pop("collection_name")
                vs["config"] = vs_cfg
                normalized["vector_store"] = vs
            return _deep_merge(base, normalized)
        return base

    def _maybe_reset_local_store(self, mem_config: dict) -> None:
        """Allow explicit reset/migration path for local Mem0 vector store."""
        reset = os.getenv("MEM0_RESET_LOCAL_STORE", "").lower() in {"1", "true", "yes"}
        if not reset:
            return

        try:
            path = Path(mem_config["vector_store"]["config"]["path"])
        except Exception:
            return

        if path.exists() and path.name in {"qdrant", ".mem0", "mem0"}:
            shutil.rmtree(path, ignore_errors=True)
            logger.warning("MEM0_RESET_LOCAL_STORE enabled: reset local Mem0 vector store at %s", path)

    def _warn_once(self, key: str, message: str) -> None:
        warned = getattr(self, "_warned_errors", set())
        if key in warned:
            return
        warned.add(key)
        self._warned_errors = warned
        logger.warning(message)

    def _degrade_if_fatal(self, exc: Exception) -> None:
        msg = str(exc).lower()
        fatal_markers = [
            "not aligned",
            "validationerror",
            "missing",
            "dimension",
        ]
        if any(marker in msg for marker in fatal_markers):
            self._enabled = False
            self._warn_once("disabled", "Mem0 disabled for current run due to persistent configuration/runtime error")

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def _call_client(self, fn, *args, **kwargs):
        return await asyncio.to_thread(partial(fn, *args, **kwargs))

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
            result = await self._call_client(self._client.add,
                messages,
                user_id=self.PIPELINE_USER_ID,
                agent_id=self.AGENT_ID,
                run_id=run_id or "store_article",
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
        except Exception as exc:
            self._warn_once("store_article", f"Mem0 store_article degraded: {exc}")
            self._degrade_if_fatal(exc)
            logger.debug("store_article failure for %s", article_id, exc_info=True)
            return None

    async def find_similar_articles(
        self,
        title: str,
        content: str = "",
        limit: int = 5,
        run_id: str = "",
    ) -> list[dict]:
        """Search memory for previously seen articles similar to the query.

        Returns a list of dicts with keys: article_id, score, memory.
        """
        if not self._enabled:
            return []

        query = f"{title}. {content[:300]}" if content else title

        try:
            results = await self._call_client(self._client.search,
                query,
                user_id=self.PIPELINE_USER_ID,
                agent_id=self.AGENT_ID,
                run_id=run_id or "find_similar_articles",
                limit=limit,
            )
            return _normalize_search_results(results)
        except Exception as exc:
            self._warn_once("find_similar_articles", f"Mem0 similarity search degraded: {exc}")
            self._degrade_if_fatal(exc)
            logger.debug("find_similar_articles failure", exc_info=True)
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
            result = await self._call_client(self._client.add,
                messages,
                user_id=self.EDITORIAL_USER_ID,
                agent_id=self.AGENT_ID,
                run_id=run_id or "store_decision",
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
        except Exception as exc:
            self._warn_once("store_decision", f"Mem0 store_decision degraded: {exc}")
            self._degrade_if_fatal(exc)
            logger.debug("store_decision failure for %s", article_id, exc_info=True)
            return None

    async def get_past_decisions(
        self,
        query: str = "editorial decisions",
        limit: int = 10,
        run_id: str = "",
    ) -> list[dict]:
        """Retrieve past editorial decisions relevant to a query."""
        if not self._enabled:
            return []

        try:
            results = await self._call_client(self._client.search,
                query,
                user_id=self.EDITORIAL_USER_ID,
                agent_id=self.AGENT_ID,
                run_id=run_id or "get_past_decisions",
                limit=limit,
            )
            return _normalize_search_results(results)
        except Exception as exc:
            self._warn_once("get_past_decisions", f"Mem0 get_past_decisions degraded: {exc}")
            self._degrade_if_fatal(exc)
            logger.debug("get_past_decisions failure", exc_info=True)
            return []

    async def get_all_memories(self, user_id: str = PIPELINE_USER_ID) -> list[dict]:
        """Retrieve all stored memories for a given user scope."""
        if not self._enabled:
            return []

        try:
            result = await self._call_client(self._client.get_all, user_id=user_id)
            if isinstance(result, dict):
                return result.get("results", result.get("memories", []))
            if isinstance(result, list):
                return result
            return []
        except Exception as exc:
            self._warn_once("get_all_memories", f"Mem0 get_all_memories degraded: {exc}")
            self._degrade_if_fatal(exc)
            logger.debug("get_all_memories failure", exc_info=True)
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


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
