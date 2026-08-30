"""
watchdog.py -- a SEPARATE process that watches the engine's heartbeat.

This is deliberately the dumbest, most bulletproof thing in the whole
project. It has exactly one power: creating the kill file. It never talks
to Alpaca and never trades. The idea is that even if engine.py hangs
(deadlocks, gets stuck in a blocking network call forever, whatever), a
process that is watching a timestamp from the outside can still stop it.

Uses ONLY the standard library, on purpose -- this must be runnable in an
air-gapped or minimal environment with no pip install required, since it's
your last line of defense if the main engine process wedges. notify.py is
pure stdlib too (smtplib/ssl/json), so calling it here to report a trip
doesn't compromise that -- it's a side effect of the one power this file
already has (creating the kill file), not a new one.

Usage: `python watchdog.py` (reads config the same way engine does, via
config.load_config(), so it agrees with the engine about file paths and
thresholds without needing separate flags).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def _read_state_timestamp(state_file_path: str):
    """Return the `written_at` timestamp (as a time.time()-comparable float
    epoch seconds) from runtime_state.json, or None if the file is missing,
    unreadable, or malformed. Any of those cases means "we can't confirm
    the engine is alive" -- callers should treat None as staleness once the
    startup grace period has passed, not as "everything's fine."
    """
    path = Path(state_file_path)
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        written_at = data["written_at"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None

    from datetime import datetime

    try:
        dt = datetime.fromisoformat(written_at)
    except ValueError:
        return None
    return dt.timestamp()


def check_once(
    state_file_path: str,
    kill_file_path: str,
    stale_after_sec: float,
    started_at: float,
    grace_period_sec: float,
    now: float = None,
) -> bool:
    """Run one watchdog check. Returns True if a kill file was (newly)
    created this call. Pure-ish (aside from file I/O) so it's easy to unit
    test with fake clocks -- `now` lets tests avoid sleeping.

    Logic:
      - Within `grace_period_sec` of watchdog startup, never trip. This
        avoids a false trip while the engine is still starting up and
        hasn't written its first state snapshot yet.
      - After the grace period, if we can't determine a heartbeat timestamp
        at all (file missing/corrupt), OR the timestamp is older than
        `stale_after_sec`, trigger the kill file.
      - If the kill file already exists, do nothing (idempotent -- don't
        spam writes, and don't clear an operator's existing kill file).
    """
    if now is None:
        now = time.time()

    kill_path = Path(kill_file_path)
    if kill_path.exists():
        return False  # already triggered (by us or an operator) -- nothing to do

    if now - started_at < grace_period_sec:
        return False  # still within startup grace period

    heartbeat = _read_state_timestamp(state_file_path)
    if heartbeat is None or (now - heartbeat) > stale_after_sec:
        kill_path.write_text(
            "HALT\nwatchdog: heartbeat stale or missing\n", encoding="utf-8"
        )
        return True

    return False


def run_forever(
    state_file_path: str,
    kill_file_path: str,
    stale_after_sec: float = 300.0,
    grace_period_sec: float = 120.0,
    poll_interval_sec: float = 15.0,
    cfg=None,
) -> None:
    """`cfg`, if given, is used ONLY to send a best-effort notification when
    the watchdog trips (see notify.py) -- passing None (the default) keeps
    this function's core behavior identical to before notifications
    existed, which is what the unit tests exercise via check_once()."""
    started_at = time.time()
    print(
        f"[watchdog] watching {state_file_path!r} (stale after {stale_after_sec}s, "
        f"grace period {grace_period_sec}s, polling every {poll_interval_sec}s)"
    )
    while True:
        tripped = check_once(state_file_path, kill_file_path, stale_after_sec, started_at, grace_period_sec)
        if tripped:
            print(f"[watchdog] heartbeat stale -- created kill file at {kill_file_path!r}")
            if cfg is not None:
                import notify

                notify.send_notification(
                    cfg, "Watchdog HALTED the engine: stale heartbeat",
                    f"runtime_state.json ({state_file_path}) hasn't been updated recently -- the "
                    f"engine may be hung or crashed. The watchdog created the kill file "
                    f"({kill_file_path}) as a precaution. Check the engine's error log.",
                )
            # Once tripped, keep running (so we can report it), but there is
            # nothing more to check until the operator clears the kill file
            # and restarts things.
        time.sleep(poll_interval_sec)


if __name__ == "__main__":
    from config import load_config

    cfg = load_config()
    # The watchdog's staleness threshold is deliberately a few multiples of
    # the engine's own loop interval, so normal tick-to-tick timing jitter
    # never trips it -- only a genuinely hung engine does.
    stale_after = max(300.0, cfg.loop_interval_sec * 5)
    try:
        run_forever(cfg.state_file_path, cfg.kill_file_path, stale_after_sec=stale_after, cfg=cfg)
    except KeyboardInterrupt:
        print("[watchdog] stopped")
        sys.exit(0)
