"""
sleeve_history.py -- one line per day of every strategy sleeve's value.

The Compare tab charts each sleeve over the whole experiment (months), so
this keeps exactly one record per UTC day -- the latest tick's numbers
overwrite that day's line -- and never prunes. That stays tiny (a few
hundred bytes a day) where equity_history.jsonl, which appends every tick
and keeps only 35 days, could not serve a months-long chart.

Record shape:
  {"date": "2026-10-05", "ts": <iso>, "sleeves": {
      "core": {"equity": 501.2, "cash": 3.1, "benchmark": null},
      "sma":  {"equity": 498.7, "cash": 410.0, "benchmark": 503.9}, ...}}

Display-only, like equity_history.py: nothing in the trading logic reads
it, and a write failure is printed and swallowed, never raised.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

SLEEVE_HISTORY_FILE = "sleeve_history.jsonl"


def record_today(file_path: str, sleeves: Dict[str, Dict]) -> None:
    """Upsert today's line with `sleeves` (sleeve id -> {equity, cash, benchmark})."""
    now = datetime.now(timezone.utc)
    today = now.date().isoformat()
    record = {"date": today, "ts": now.isoformat(), "sleeves": sleeves}
    try:
        kept = [e for e in read_all(file_path) if e.get("date") != today]
        kept.append(record)
        kept.sort(key=lambda e: str(e.get("date", "")))
        path = Path(file_path)
        tmp_path = path.with_name(path.name + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            for e in kept:
                f.write(json.dumps(e) + "\n")
        tmp_path.replace(path)
    except OSError as e:
        print(f"[sleeve_history] failed to record today's snapshot: {e}", file=sys.stderr)


def read_all(file_path: str) -> List[dict]:
    """Every daily record, oldest first. Never raises."""
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
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(entry, dict) and isinstance(entry.get("sleeves"), dict):
                    entries.append(entry)
    except OSError:
        return []
    return sorted(entries, key=lambda e: str(e.get("date", "")))
