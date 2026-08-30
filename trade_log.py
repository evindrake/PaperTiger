"""
trade_log.py -- a small, append-only, durable record of every order this
bot ever submitted (tactical AND core-satellite bootstrap), independent of
the rolling in-memory event log (capped at 200 entries, reset on restart)
and independent of whatever the broker's own order history shows.

Why this exists: runtime_state.json's event log is a rolling buffer for
the dashboard, not a permanent record -- restart the engine and it's gone.
Alpaca's own order history IS durable, but living entirely inside the
broker makes later analysis (e.g. "how many trades did the RSI signal make
last month, and why") slower and more awkward than a local file you can
grep, tail, or load into a spreadsheet with one line of pandas.

Nothing here is read by any trading logic -- this module is write-only
from the strategy/engine's point of view, one JSON object per line
(JSONL), so it's trivial to append to and to stream/tail without parsing
the whole file.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from typing import Optional

TRADE_LOG_FILE = "trade_history.jsonl"


def append_trade(
    file_path: str,
    source: str,
    symbol: str,
    side: str,
    reason: str,
    qty: Optional[float] = None,
    notional_usd: Optional[float] = None,
    limit_price: Optional[float] = None,
    order_id: Optional[str] = None,
    client_order_id: Optional[str] = None,
) -> None:
    """Append one JSON line describing a submitted order.

    `source` is "tactical" (the signal-driven strategy) or "core_bootstrap"
    (the one-time core-satellite allocation, see core.py).

    Best-effort: a logging failure here must never interrupt a trading
    decision -- any OSError is printed and swallowed, never raised.
    """
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "notional_usd": notional_usd,
        "limit_price": limit_price,
        "order_id": order_id,
        "client_order_id": client_order_id,
        "reason": reason,
    }
    try:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as e:
        print(f"[trade_log] failed to append trade record: {e}", file=sys.stderr)
