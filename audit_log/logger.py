from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

AUDIT_DIR = Path(__file__).resolve().parent


def write_audit_entry(run_id: str, event: str, **data: object) -> None:
    """Append a single NDJSON audit entry to the run's log file."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "event": event,
        **data,
    }
    path = AUDIT_DIR / f"run_{run_id}.ndjson"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def read_audit_log(run_id: str) -> list[dict]:
    """Read all entries from a run's audit log."""
    path = AUDIT_DIR / f"run_{run_id}.ndjson"
    if not path.exists():
        return []
    entries = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries
