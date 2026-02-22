from __future__ import annotations

from click.testing import CliRunner

from scripts import cli


def test_cli_run_single_feature_override_true(monkeypatch):
    called: dict[str, object] = {}

    async def fake_run_pipeline(single_feature_mode=None):  # noqa: ANN001
        called["single_feature_mode"] = single_feature_mode

        class DummyContext:
            newsletter = None

        return DummyContext()

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    runner = CliRunner()
    result = runner.invoke(cli.main, ["run", "--single-feature"])

    assert result.exit_code == 0
    assert called["single_feature_mode"] is True


def test_cli_run_single_feature_override_false(monkeypatch):
    called: dict[str, object] = {}

    async def fake_run_pipeline(single_feature_mode=None):  # noqa: ANN001
        called["single_feature_mode"] = single_feature_mode

        class DummyContext:
            newsletter = None

        return DummyContext()

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    runner = CliRunner()
    result = runner.invoke(cli.main, ["run", "--no-single-feature"])

    assert result.exit_code == 0
    assert called["single_feature_mode"] is False
