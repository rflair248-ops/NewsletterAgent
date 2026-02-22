from __future__ import annotations

import pytest

from engine.agents.editor import EditorAgent
from engine.agents.reviewer import ReviewerAgent
from engine.llm_router import _validate_cli_model
from models.article import Article
from models.enums import ArticleStatus, ContentCategory
from models.newsletter import Newsletter
from policy.quality_checks import CRITIC_PROMPT, _count_tactical_moves


def test_critic_prompt_format_no_crash():
    rendered = CRITIC_PROMPT.format(keyword="k", source_urls_json="[]", markdown="# x")
    assert '"pass"' in rendered


def test_count_tactical_moves_handles_multidigit():
    md = "## 3 Tactical Moves\n1. One\n2. Two\n10. Ten"
    assert _count_tactical_moves(md) == 3


def test_validate_cli_model():
    assert _validate_cli_model("sonnet") == "sonnet"
    with pytest.raises(ValueError):
        _validate_cli_model("bad model!")


@pytest.mark.asyncio
async def test_reviewer_uses_markdown_body_in_single_feature(pipeline_context, monkeypatch):
    pipeline_context.newsletter = Newsletter(
        edition_id="x",
        subject_line="x",
        sections=[],
        total_articles=1,
        markdown_body="# Body\n\ncontent",
        metadata={"mode": "single_feature"},
    )
    pipeline_context.settings["llm"]["local"]["provider"] = "claude_cli"

    captured = {}

    async def fake_cli(prompt: str, system: str, model: str):
        captured["prompt"] = prompt
        return '{"approved": true, "issues": [], "suggestions": []}'

    monkeypatch.setattr("engine.agents.reviewer.claude_cli_completion", fake_cli)
    agent = ReviewerAgent(pipeline_context)
    await agent.run()

    assert "# Body" in captured["prompt"]


@pytest.mark.asyncio
async def test_editor_persists_memory_on_review_pass(pipeline_context, monkeypatch):
    pipeline_context.settings["pipeline"]["single_feature_mode"] = True
    a = Article(id="a1", url="https://a.com/1", title="A", source_name="S", category=ContentCategory.BUSINESS, status=ArticleStatus.DEDUPLICATED)
    pipeline_context.curated_articles = [a]
    pipeline_context.newsletter = Newsletter(edition_id="x", subject_line="x", sections=[], total_articles=1, markdown_body="# Hi", metadata={"mode": "single_feature", "target_keyword": "k"})
    pipeline_context.review_result = {"approved": True, "issues": []}

    calls = {"remember": 0}

    async def fake_remember(article):
        calls["remember"] += 1

    async def fake_critic(markdown, keyword, source_urls, model):
        return {"pass": True, "actionable_fixes": []}

    monkeypatch.setattr("engine.agents.editor.critic_scorecard", fake_critic)
    agent = EditorAgent(pipeline_context)
    agent._remember_article = fake_remember  # type: ignore[method-assign]
    await agent.run()

    assert calls["remember"] == 1
