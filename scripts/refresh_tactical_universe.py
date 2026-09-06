"""
refresh_tactical_universe.py -- weekly job that (re)selects the dynamic
satellite/tactical symbol pool the engine trades IN ADDITION TO the static
core whitelist (cfg.symbols).

Deliberately NOT a full scan of the US equity market: that would be slow,
API-heavy, and pointless at this account size. Instead:

  1. Load candidate_universe.json -- a small, static, user-editable pool of
     well-known liquid US stocks (the "eligible" list, never traded directly).
  2. One bulk call to Alpaca (broker.list_tradable_us_equities()) to confirm
     which candidates are CURRENTLY active + tradable + a plain US equity
     (not OTC/inactive/delisted since the file was last hand-edited).
  3. For the survivors, pull cfg.tactical_universe_lookback_days of daily
     bars and rank by average dollar volume (avg_close * avg_volume) --
     picking liquid, easily-fillable names, not just "large index members."
  4. Write the top cfg.tactical_universe_size of them to
     tactical_universe.json (atomic tmp+rename), which engine.py reads
     fresh every tick and merges on top of cfg.symbols -- core.py's
     CoreAllocator never sees this file, so core bootstrap sizing is
     unaffected by anything this script does.

Run this on a schedule (weekly is plenty -- see scripts/install_services.ps1
for the Windows Task Scheduler registration); it is safe to run more often
or by hand, it's fully idempotent and read-mostly against the broker.

Usage:
    python scripts/refresh_tactical_universe.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from broker import Broker, BrokerError
from config import load_config


def load_candidate_universe(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    symbols = raw.get("symbols") if isinstance(raw, dict) else None
    if not isinstance(symbols, list):
        return []
    return [s.strip().upper() for s in symbols if isinstance(s, str) and s.strip()]


def rank_by_dollar_volume(broker: Broker, symbols: List[str], lookback_days: int) -> List[tuple]:
    """Returns [(symbol, avg_dollar_volume), ...] sorted descending. Skips
    (rather than crashes on) any symbol whose bars fail to fetch or have no
    usable history -- one bad candidate should never abort the whole
    refresh."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days * 2)  # buffer for weekends/holidays
    ranked = []
    for symbol in symbols:
        try:
            bars = broker.bars(symbol, start, end)
        except BrokerError as e:
            print(f"[refresh_tactical_universe] skipping {symbol}: {e}")
            continue
        bars = bars[-lookback_days:]
        if not bars:
            continue
        avg_dollar_volume = sum(b.close * b.volume for b in bars) / len(bars)
        if avg_dollar_volume > 0:
            ranked.append((symbol, avg_dollar_volume))
    ranked.sort(key=lambda pair: pair[1], reverse=True)
    return ranked


def write_tactical_universe(path: str, symbols: List[str], metadata: dict) -> None:
    payload = {"symbols": symbols, "generated_at": datetime.now(timezone.utc).isoformat(), **metadata}
    p = Path(path)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(p)


def main() -> None:
    cfg = load_config()
    broker = Broker(cfg)

    candidates = load_candidate_universe(cfg.candidate_universe_file_path)
    if not candidates:
        print(f"[refresh_tactical_universe] no candidates in {cfg.candidate_universe_file_path} -- nothing to do")
        write_tactical_universe(cfg.tactical_universe_file_path, [], {"candidates_considered": 0})
        return

    try:
        tradable_assets = broker.list_tradable_us_equities()
    except BrokerError as e:
        print(f"[refresh_tactical_universe] could not list tradable assets, aborting: {e}", file=sys.stderr)
        raise SystemExit(1)

    tradable_by_symbol = {a.symbol: a for a in tradable_assets}
    survivors = [
        symbol for symbol in candidates
        if tradable_by_symbol.get(symbol) is not None and tradable_by_symbol[symbol].tradable
    ]
    dropped = sorted(set(candidates) - set(survivors))
    if dropped:
        print(f"[refresh_tactical_universe] dropping candidates not currently active/tradable: {dropped}")

    ranked = rank_by_dollar_volume(broker, survivors, cfg.tactical_universe_lookback_days)
    selected = [symbol for symbol, _ in ranked[: cfg.tactical_universe_size]]

    write_tactical_universe(
        cfg.tactical_universe_file_path,
        selected,
        {
            "candidates_considered": len(candidates),
            "tradable_survivors": len(survivors),
            "lookback_days": cfg.tactical_universe_lookback_days,
        },
    )
    print(f"[refresh_tactical_universe] wrote {len(selected)} symbols to {cfg.tactical_universe_file_path}: {selected}")


if __name__ == "__main__":
    main()
