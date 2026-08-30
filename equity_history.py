"""
equity_history.py -- a small, self-pruning JSONL log of the live account's
value over time, so the dashboard's Live tab can show a "last 30 days"
performance chart instead of only a single current snapshot.

Why this needs its own file rather than reusing runtime_state.json: that
file is overwritten every tick with only the CURRENT snapshot -- it has no
memory of what equity looked like an hour or a day ago. This module gives
it one, at low cost: one entry is appended per engine tick (every
LOOP_INTERVAL_SEC, 300s by default), and entries older than
`retention_days` are dropped on every write so the file stays small and
bounded without a separate cleanup job.

Nothing here is read by any trading logic -- like trade_log.py, this
module is write-only from the engine's point of view; only dashboard.py
reads it back, purely for display.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

EQUITY_HISTORY_FILE = "equity_history.jsonl"

# 30 days of chart + a little slack, so "last 30 days" never comes up short
# by a few hours due to when exactly the retention prune last ran.
DEFAULT_RETENTION_DAYS = 35


def append_snapshot(
    file_path: str,
    equity: float,
    cash: float,
    positions_value: Dict[str, float],
    retention_days: int = DEFAULT_RETENTION_DAYS,
) -> None:
    """Append one snapshot and drop anything older than `retention_days`.

    Best-effort, like trade_log.py/notify.py: a logging failure here must
    never interrupt a trading decision -- any failure is printed and
    swallowed, never raised.
    """
    now = datetime.now(timezone.utc)
    record = {
        "ts": now.isoformat(),
        "equity": equity,
        "cash": cash,
        "positions_value": sum(positions_value.values()) if positions_value else 0.0,
        "positions": dict(positions_value),
    }
    try:
        existing = _read_all(file_path)
        cutoff = (now - timedelta(days=retention_days)).isoformat()
        kept = [e for e in existing if str(e.get("ts", "")) >= cutoff]
        kept.append(record)

        path = Path(file_path)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            for e in kept:
                f.write(json.dumps(e) + "\n")
        tmp_path.replace(path)
    except OSError as e:
        print(f"[equity_history] failed to append snapshot: {e}", file=sys.stderr)


def read_recent(file_path: str, days: int = 30) -> List[dict]:
    """Return snapshots from the last `days` days, oldest first. Never
    raises -- a missing/corrupt file just yields an empty list."""
    entries = _read_all(file_path)
    if not entries:
        return entries
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    recent = [e for e in entries if str(e.get("ts", "")) >= cutoff]
    return sorted(recent, key=lambda e: str(e.get("ts", "")))


def _read_all(file_path: str) -> List[dict]:
    path = Path(file_path)
    if not path.exists():
        return []
    entries: List[dict] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # skip a corrupted/partially-written line, don't fail the whole read
    except OSError:
        return []
    return entries
