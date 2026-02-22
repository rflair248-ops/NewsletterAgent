from __future__ import annotations

import asyncio
import os
import re
from typing import Any

import httpx

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


async def local_completion(
    model: str,
    prompt: str,
    system: str = "",
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> str:
    """Call a local Ollama model for completions."""
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")


async def local_chat(
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.3,
) -> str:
    """Call a local Ollama model with chat format."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")


def _validate_cli_model(model: str) -> str:
    candidate = (model or "").strip()
    if not candidate:
        raise ValueError("Claude CLI model must be a non-empty string")
    if not re.match(r"^[a-zA-Z0-9._:-]+$", candidate):
        raise ValueError(f"Invalid Claude CLI model value: {candidate!r}")
    return candidate


async def claude_cli_completion(
    prompt: str,
    system: str = "",
    model: str = "sonnet",
    timeout_seconds: float = 180.0,
) -> str:
    """Use Claude Code CLI authenticated session (no API key in code path)."""
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    cli_model = _validate_cli_model(model)
    proc = await asyncio.create_subprocess_exec(
        "claude",
        "-p",
        "--model",
        cli_model,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(
            proc.communicate(input=full_prompt.encode("utf-8")),
            timeout=timeout_seconds,
        )
    except TimeoutError as exc:
        proc.kill()
        await proc.wait()
        raise RuntimeError(f"claude CLI timed out after {timeout_seconds:.1f}s") from exc

    if proc.returncode != 0:
        raise RuntimeError(f"claude CLI failed: {err.decode('utf-8', errors='ignore').strip()}")
    return out.decode("utf-8", errors="ignore").strip()
