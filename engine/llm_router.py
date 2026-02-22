from __future__ import annotations

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
