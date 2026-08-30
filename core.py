"""
core.py -- the "core" (permanent buy-and-hold) sleeve of a core-satellite split.

Why this exists: backtest.py and walkforward.py have shown that the
tactical signal (SMA crossover or RSI reversion) does not reliably beat
plain buy-and-hold. A core-satellite split hedges against that by permanently
setting aside a fixed fraction of capital (`cfg.core_allocation_pct`) into an
equal-weight, buy-once-and-never-sell position across the symbol whitelist.
That portion captures the market's long-run drift no matter what the
tactical signal does. The remainder ("satellite") is the only capital
strategy.py's signal is ever allowed to trade.

This is a ONE-TIME, idempotent bootstrap, not an ongoing strategy:
  - On the first tick(s) after the bot starts, ensure_core_positions()
    submits one buy per symbol sized at (core_allocation_pct * seed_usd) /
    len(symbols), and records the FILLED quantity into core_holdings.json
    once each order confirms filled.
  - After every symbol has a recorded core qty, this module is a no-op on
    every later tick -- it only ever reports how many shares are core (via
    tactical_available_qty), never buys or sells again.
  - There is NO code path anywhere in this project that sells a core
    position. The only way core capital changes hands is a human manually
    trading it outside the bot.

Core buys intentionally do NOT go through safety.PreTradeCheck's tactical
caps (max_position_usd / max_concentration_pct) -- those caps exist to
bound the small, frequent tactical trade size and would make no sense
applied to a single, larger, permanent allocation. Core buys still respect
the structural cash-account/long-only guarantee (they're plain limit buys
through the same broker.py) and a minimal set of sanity checks (whitelisted
symbol, fresh quote, sufficient buying power minus the cash buffer) -- see
_MINIMAL core check below.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import trade_log

CORE_CLIENT_ID_PREFIX = "pt-core-"


def core_client_order_id(symbol: str) -> str:
    return f"{CORE_CLIENT_ID_PREFIX}{symbol}-buy"


class CoreAllocator:
    def __init__(self, cfg, holdings_path: Optional[str] = None):
        self.cfg = cfg
        self.path = Path(holdings_path or cfg.core_holdings_file_path)

    def load(self) -> Dict[str, float]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _save(self, holdings: Dict[str, float]) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(holdings, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def is_established(self) -> bool:
        if self.cfg.core_allocation_pct <= 0:
            return True  # core-satellite disabled entirely -- nothing to establish
        holdings = self.load()
        return all(s in holdings for s in self.cfg.symbols)

    def tactical_available_qty(self, symbol: str, held_qty: float) -> float:
        """How much of a held position the TACTICAL signal is allowed to
        touch -- total held minus whatever is reserved as core. Never
        negative (a core position without a matching tactical qty on top
        just means the tactical sleeve currently holds nothing here)."""
        core_qty = self.load().get(symbol, 0.0)
        return max(0.0, held_qty - core_qty)

    def ensure_core_positions(self, broker, account, quotes, open_orders: List) -> List[str]:
        """Idempotent bootstrap step -- call this once per engine tick
        (and once per backtest bar, in spirit) before running the tactical
        strategy. Returns a list of human-readable log lines describing
        what happened, so the caller can fold them into its own event log
        without this module needing to know about engine.py's logging
        format.

        Safe to call forever: once every whitelisted symbol has a recorded
        core qty, this returns immediately with no side effects.
        """
        logs: List[str] = []
        if self.cfg.core_allocation_pct <= 0:
            return logs  # core-satellite disabled entirely (e.g. core_allocation_pct=0)

        holdings = self.load()
        missing = [s for s in self.cfg.symbols if s not in holdings]
        if not missing:
            return logs

        target_usd_per_symbol = (self.cfg.core_allocation_pct * self.cfg.seed_usd) / len(self.cfg.symbols)
        open_by_client_id = {o.client_order_id: o for o in open_orders}

        for symbol in missing:
            cid = core_client_order_id(symbol)

            if cid in open_by_client_id:
                continue  # still working -- check again next tick

            existing = broker.order_by_client_id(cid)
            if existing is not None:
                if existing.status == "filled" and existing.filled_qty:
                    holdings[symbol] = existing.filled_qty
                    self._save(holdings)
                    logs.append(f"core position established for {symbol}: {existing.filled_qty} shares")
                # else: order exists but isn't filled and isn't open (e.g.
                # rejected/canceled) -- deliberately do NOT auto-retry. A
                # rejected core buy is worth a human looking at, not a bot
                # silently resubmitting forever.
                continue

            quote = quotes.get(symbol)
            if quote is None:
                continue
            if quote.age_sec > self.cfg.quote_max_age_sec:
                continue
            available = account.buying_power - self.cfg.cash_buffer_usd
            if target_usd_per_symbol > available:
                logs.append(
                    f"core bootstrap for {symbol} deferred: need ${target_usd_per_symbol:.2f}, "
                    f"only ${available:.2f} available after cash buffer"
                )
                continue

            limit_price = quote.ask * 1.001  # small nudge above ask, same convention as strategy.py
            order = broker.submit_limit_buy(symbol, target_usd_per_symbol, limit_price, cid)
            trade_log.append_trade(
                getattr(self.cfg, "trade_log_file_path", trade_log.TRADE_LOG_FILE),
                source="core_bootstrap", symbol=symbol, side="buy",
                reason="one-time core-satellite allocation bootstrap",
                notional_usd=target_usd_per_symbol, limit_price=limit_price,
                order_id=order.id, client_order_id=cid,
            )
            logs.append(f"submitted core bootstrap buy for {symbol}: ${target_usd_per_symbol:.2f}")

        return logs
