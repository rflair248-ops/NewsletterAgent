"""Direct pipeline runner entry point with post-run self-improvement."""

from __future__ import annotations

import asyncio

from pipeline.daily_run import run_pipeline
from scripts.self_improve import main as self_improve_main


async def _run_all() -> None:
    await run_pipeline()
    await self_improve_main()


if __name__ == "__main__":
    asyncio.run(_run_all())
