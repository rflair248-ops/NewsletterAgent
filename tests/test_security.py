from __future__ import annotations

from pathlib import Path


def test_http_server_must_bind_localhost() -> None:
    """Guardrail: any `python -m http.server` usage must bind localhost only."""
    root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []

    pattern = "python -m " + "http.server"

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path == Path(__file__).resolve():
            continue
        if ".venv" in path.parts or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        if path.suffix.lower() not in {".py", ".sh", ".md", ".txt", ".yaml", ".yml"}:
            continue

        try:
            lines = path.read_text(errors="ignore").splitlines()
        except Exception:
            continue

        for i, line in enumerate(lines, start=1):
            if pattern in line and "--bind 127.0.0.1" not in line:
                offenders.append(f"{path.relative_to(root)}:{i}")

    assert not offenders, "Insecure http.server invocation(s) found: " + ", ".join(offenders)
