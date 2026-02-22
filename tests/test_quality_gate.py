from __future__ import annotations

import json

import pytest

from engine.agents.editor import EditorAgent
from models.newsletter import Newsletter


@pytest.mark.asyncio
async def test_single_feature_qa_revises_once_and_rechecks(tmp_path, monkeypatch, pipeline_context):
    monkeypatch.chdir(tmp_path)
    pipeline_context.settings["pipeline"]["single_feature_mode"] = True
    pipeline_context.newsletter = Newsletter(
        edition_id="20260222",
        subject_line="Test",
        sections=[],
        total_articles=2,
        markdown_body="# Title\n\nBody",
        metadata={"mode": "single_feature", "target_keyword": "legal AI strategy"},
    )

    agent = EditorAgent(pipeline_context)
    agent._remember_article = lambda article: None  # type: ignore[method-assign]

    calls = {"critic": 0, "revise": 0}

    async def fake_critic(markdown, keyword, source_urls, model="sonnet"):  # noqa: ANN001
        calls["critic"] += 1
        if calls["critic"] == 1:
            return {"pass": False, "actionable_fixes": ["Add keyword in first 100 words"]}
        return {"pass": True, "actionable_fixes": []}

    async def fake_revise(markdown, keyword, fixes, model="sonnet"):  # noqa: ANN001
        calls["revise"] += 1
        return markdown + "\n\nlegal AI strategy appears early."

    monkeypatch.setattr("engine.agents.editor.critic_scorecard", fake_critic)
    monkeypatch.setattr("engine.agents.editor.revise_with_fixes", fake_revise)

    await agent.run()

    assert calls["critic"] == 2
    assert calls["revise"] == 1

    checks_path = tmp_path / "output" / "quality_checks.jsonl"
    lines = checks_path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["pass"] is False
    assert json.loads(lines[1])["pass"] is True


@pytest.mark.asyncio
async def test_single_feature_qa_falls_back_when_critic_unavailable(tmp_path, monkeypatch, pipeline_context):
    monkeypatch.chdir(tmp_path)
    pipeline_context.settings["pipeline"]["single_feature_mode"] = True
    pipeline_context.newsletter = Newsletter(
        edition_id="20260222",
        subject_line="Test",
        sections=[],
        total_articles=2,
        markdown_body="# Strong claim\n\nlegal AI strategy in opening words.\n\n## Trigger\nA\n## Translation\nB\n## 3 Tactical Moves\n1. A\n2. B\n3. C\n\nLook ahead.",
        metadata={"mode": "single_feature", "target_keyword": "legal AI strategy"},
    )

    agent = EditorAgent(pipeline_context)
    agent._remember_article = lambda article: None  # type: ignore[method-assign]

    async def boom(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("claude missing")

    monkeypatch.setattr("engine.agents.editor.critic_scorecard", boom)

    await agent.run()

    checks_path = tmp_path / "output" / "quality_checks.jsonl"
    assert checks_path.exists()
    payload = json.loads(checks_path.read_text().strip().splitlines()[0])
    assert payload["critic_mode"] == "rule_based"
    assert payload.get("warning") == "critic_unavailable_rule_based_fallback"
