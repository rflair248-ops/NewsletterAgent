from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memory.mem0_store import Mem0Store, _extract_memory_id, _normalize_search_results


class TestMem0StoreDisabled:
    """Tests for Mem0Store when mem0ai is not installed or no key is set."""

    def test_disabled_without_package(self):
        with patch.dict("os.environ", {}, clear=True):
            store = Mem0Store(api_key=None, config={"_skip_init": True})
        # Should gracefully disable if mem0 can't be imported
        # (In test env it likely can't, so _enabled should be False)

    def test_enabled_property(self):
        store = Mem0Store.__new__(Mem0Store)
        store._enabled = False
        store._client = None
        assert store.enabled is False

    @pytest.mark.asyncio
    async def test_store_article_noop_when_disabled(self):
        store = Mem0Store.__new__(Mem0Store)
        store._enabled = False
        store._client = None
        result = await store.store_article("id1", "Title", "Source")
        assert result is None

    @pytest.mark.asyncio
    async def test_find_similar_noop_when_disabled(self):
        store = Mem0Store.__new__(Mem0Store)
        store._enabled = False
        store._client = None
        result = await store.find_similar_articles("Some title")
        assert result == []

    @pytest.mark.asyncio
    async def test_store_decision_noop_when_disabled(self):
        store = Mem0Store.__new__(Mem0Store)
        store._enabled = False
        store._client = None
        result = await store.store_decision("run1", "art1", "published")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_past_decisions_noop_when_disabled(self):
        store = Mem0Store.__new__(Mem0Store)
        store._enabled = False
        store._client = None
        result = await store.get_past_decisions()
        assert result == []

    @pytest.mark.asyncio
    async def test_get_all_memories_noop_when_disabled(self):
        store = Mem0Store.__new__(Mem0Store)
        store._enabled = False
        store._client = None
        result = await store.get_all_memories()
        assert result == []


class TestMem0StoreEnabled:
    """Tests for Mem0Store with a mocked mem0 client."""

    def _make_store(self) -> Mem0Store:
        store = Mem0Store.__new__(Mem0Store)
        store.api_key = "test-key"
        store._enabled = True
        store._client = MagicMock()
        return store

    @pytest.mark.asyncio
    async def test_store_article_success(self):
        store = self._make_store()
        store._client.add.return_value = {"results": [{"id": "mem_abc"}]}

        result = await store.store_article(
            article_id="art1",
            title="Test Article",
            source="Test Source",
            summary="A summary",
            run_id="run1",
        )

        assert result == "mem_abc"
        store._client.add.assert_called_once()
        call_kwargs = store._client.add.call_args
        assert call_kwargs[1]["user_id"] == "newsletter_pipeline"
        assert call_kwargs[1]["agent_id"] == "newsletter_agent"
        assert call_kwargs[1]["run_id"] == "run1"
        assert call_kwargs[1]["metadata"]["article_id"] == "art1"

    @pytest.mark.asyncio
    async def test_store_article_handles_exception(self):
        store = self._make_store()
        store._client.add.side_effect = RuntimeError("API error")

        result = await store.store_article("art1", "Title", "Source")
        assert result is None

    @pytest.mark.asyncio
    async def test_find_similar_articles(self):
        store = self._make_store()
        store._client.search.return_value = {
            "results": [
                {"id": "mem1", "memory": "Article about AI", "score": 0.92, "metadata": {}},
                {"id": "mem2", "memory": "Article about ML", "score": 0.78, "metadata": {}},
            ]
        }

        results = await store.find_similar_articles("AI breakthrough", content="details")

        assert len(results) == 2
        assert results[0]["memory_id"] == "mem1"
        assert results[0]["score"] == 0.92
        store._client.search.assert_called_once()
        search_kwargs = store._client.search.call_args.kwargs
        assert search_kwargs["user_id"] == "newsletter_pipeline"
        assert search_kwargs["agent_id"] == "newsletter_agent"

    @pytest.mark.asyncio
    async def test_find_similar_handles_exception(self):
        store = self._make_store()
        store._client.search.side_effect = RuntimeError("Search failed")

        results = await store.find_similar_articles("query")
        assert results == []

    @pytest.mark.asyncio
    async def test_store_decision_success(self):
        store = self._make_store()
        store._client.add.return_value = {"results": [{"id": "dec_xyz"}]}

        result = await store.store_decision("run1", "art1", "published", reason="Good article")

        assert result == "dec_xyz"
        call_kwargs = store._client.add.call_args
        assert call_kwargs[1]["user_id"] == "newsletter_editorial"
        assert call_kwargs[1]["agent_id"] == "newsletter_agent"
        assert call_kwargs[1]["run_id"] == "run1"
        assert call_kwargs[1]["metadata"]["decision"] == "published"

    @pytest.mark.asyncio
    async def test_get_past_decisions(self):
        store = self._make_store()
        store._client.search.return_value = [
            {"id": "d1", "memory": "Published article X", "score": 0.9, "metadata": {}}
        ]

        results = await store.get_past_decisions("article X")
        assert len(results) == 1
        search_kwargs = store._client.search.call_args.kwargs
        assert search_kwargs["user_id"] == "newsletter_editorial"
        assert search_kwargs["agent_id"] == "newsletter_agent"

    @pytest.mark.asyncio
    async def test_get_all_memories_dict_response(self):
        store = self._make_store()
        store._client.get_all.return_value = {"results": [{"id": "m1"}, {"id": "m2"}]}

        results = await store.get_all_memories()
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_get_all_memories_list_response(self):
        store = self._make_store()
        store._client.get_all.return_value = [{"id": "m1"}]

        results = await store.get_all_memories()
        assert len(results) == 1


class TestHelpers:
    """Tests for module-level helper functions."""

    def test_extract_memory_id_from_results_dict(self):
        assert _extract_memory_id({"results": [{"id": "abc"}]}) == "abc"

    def test_extract_memory_id_from_flat_dict(self):
        assert _extract_memory_id({"id": "xyz"}) == "xyz"

    def test_extract_memory_id_from_list(self):
        assert _extract_memory_id([{"id": "lst1"}]) == "lst1"

    def test_extract_memory_id_none(self):
        assert _extract_memory_id({}) is None
        assert _extract_memory_id(None) is None

    def test_normalize_search_results_dict_format(self):
        results = {"results": [
            {"id": "m1", "memory": "text1", "score": 0.9, "metadata": {"k": "v"}},
        ]}
        normalized = _normalize_search_results(results)
        assert len(normalized) == 1
        assert normalized[0]["memory_id"] == "m1"
        assert normalized[0]["memory"] == "text1"
        assert normalized[0]["score"] == 0.9

    def test_normalize_search_results_list_format(self):
        results = [
            {"id": "m1", "text": "text1", "relevance": 0.85, "metadata": {}},
        ]
        normalized = _normalize_search_results(results)
        assert len(normalized) == 1
        assert normalized[0]["memory"] == "text1"
        assert normalized[0]["score"] == 0.85

    def test_normalize_search_results_empty(self):
        assert _normalize_search_results([]) == []
        assert _normalize_search_results({"results": []}) == []
