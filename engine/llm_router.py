from __future__ import annotations

import asyncio
import os
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


async def claude_cli_completion(
    prompt: str,
    system: str = "",
    model: str = "sonnet",
) -> str:
    """Use Claude Code CLI authenticated session (no API key in code path)."""
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    proc = await asyncio.create_subprocess_exec(
        "claude",
        "-p",
        "--model",
        model,
        full_prompt,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"claude CLI failed: {err.decode('utf-8', errors='ignore').strip()}")
    return out.decode("utf-8", errors="ignore").strip()
