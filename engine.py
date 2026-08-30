"""
engine.py -- the live trading loop.

This is the one place where every other module gets wired together against
a real (paper, by default) broker connection. The loop is intentionally
linear and boring -- read the tick() method top to bottom and it tells you
the entire safety story:

    1. Is the kill file present?            -> obey it, do nothing new.
    2. Pull truth from the broker.           -> broker is always authoritative.
    3. Reconcile against our own records.    -> a surprise order means HALT.
    4. Run circuit breakers on the numbers.  -> may escalate to FLATTEN/HALT.
    5. If the market's open, refresh history and ask strategy to propose.
    6. Validate every proposal through PreTradeCheck.
    7. Submit only the survivors, with an idempotency key.
    8. Write a runtime_state.json snapshot, no matter what happened.

Any unhandled exception anywhere in a tick is caught at the very top level
and treated as an error tick -- feeding the consecutive-error circuit
breaker, which eventually HALTs. We never let "something went wrong that we
didn't anticipate" turn into "so let's try again immediately."
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from collections import deque
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Deque, Dict, List, Optional

import equity_history
import notify
import trade_log
from broker import Broker, BrokerError, OrderView
from core import CoreAllocator, core_client_order_id
from safety import BreakerAction, CircuitBreakers, KillMode, KillSwitch, PreTradeCheck
from strategy import OrderIntent, propose

# How many recent events to keep in the runtime-state snapshot. This is a
# rolling log for the dashboard, not an audit trail -- the broker itself is
# the durable record of what actually happened to your money.
_EVENT_LOG_MAXLEN = 200


class Engine:
    def __init__(self, cfg, broker: Optional[Broker] = None):
        self.cfg = cfg
        self.broker = broker or Broker(cfg)
        self.kill_switch = KillSwitch(cfg.kill_file_path)
        self.circuit_breakers = CircuitBreakers(cfg)
        self.pre_trade = PreTradeCheck(cfg)
        self.core = CoreAllocator(cfg)

        # symbol -> cached completed daily closes (refreshed once per day)
        self._history_cache: Dict[str, List[float]] = {}
        self._history_cache_date: Optional[date] = None

        self._events: Deque[dict] = deque(maxlen=_EVENT_LOG_MAXLEN)
        self._halted = False  # recomputed every tick in _tick_inner(); reflects
        # *this tick's* outcome, not a permanent latch -- the kill-switch file
        # (checked independently each tick) is what actually gates trading.
        self._market_open_prev: Optional[bool] = None  # None until the first
        # tick observes it -- lets us log only on open<->closed transitions
        # instead of every 5-minute tick overnight/weekends.

    # -- event log / state snapshot ----------------------------------------

    def _log(self, level: str, message: str, **extra) -> None:
        self._events.append(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "level": level,
                "message": message,
                **extra,
            }
        )

    def _write_state(
        self,
        account=None,
        positions=None,
        open_orders=None,
        kill_mode: Optional[KillMode] = None,
        core_holdings: Optional[Dict[str, float]] = None,
    ) -> None:
        """Write runtime_state.json. Called at the end of every tick, even
        an error tick, so the watchdog and dashboard always see a fresh
        timestamp reflecting "the engine is alive and this is what it saw."
        """
        equity = account.equity if account is not None else None
        if account is not None:
            equity_history.append_snapshot(
                self.cfg.equity_history_file_path,
                equity=account.equity,
                cash=account.cash,
                positions_value={sym: p.market_value for sym, p in (positions or {}).items()},
            )
        state = {
            "written_at": datetime.now(timezone.utc).isoformat(),
            "halted": self._halted,
            "kill_mode": kill_mode.value if kill_mode else None,
            "account": asdict(account) if account is not None else None,
            "positions": {sym: asdict(p) for sym, p in (positions or {}).items()},
            "open_orders": [asdict(o) for o in (open_orders or [])],
            "core_holdings": core_holdings or self.core.load(),
            "core_allocation_pct": self.cfg.core_allocation_pct,
            "daily_pl_pct": self.circuit_breakers.daily_pl_pct(equity) if equity is not None else None,
            "drawdown_pct": self.circuit_breakers.drawdown_pct(equity) if equity is not None else None,
            "circuit_breakers": self.circuit_breakers.to_dict(),
            "events": list(self._events),
            # A read-only snapshot of the operationally-relevant config, for
            # the dashboard's Status tab -- no credentials, just what's
            # actually running right now (loop cadence, signal params, caps).
            "config_snapshot": {
                "symbols": list(self.cfg.symbols),
                "signal_kind": self.cfg.signal_kind,
                "signal_fast": self.cfg.signal_fast,
                "signal_slow": self.cfg.signal_slow,
                "signal_period": self.cfg.signal_period,
                "signal_oversold": self.cfg.signal_oversold,
                "signal_overbought": self.cfg.signal_overbought,
                "target_trade_usd": self.cfg.target_trade_usd,
                "max_position_usd": self.cfg.max_position_usd,
                "max_concentration_pct": self.cfg.max_concentration_pct,
                "daily_loss_limit_pct": self.cfg.daily_loss_limit_pct,
                "max_drawdown_pct": self.cfg.max_drawdown_pct,
                "loop_interval_sec": self.cfg.loop_interval_sec,
            },
        }
        path = Path(self.cfg.state_file_path)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        # Write-then-rename so the dashboard/watchdog never see a half-written file.
        tmp_path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
        tmp_path.replace(path)

    # -- reconciliation -------------------------------------------------------

    def _reconcile(self, open_orders: List[OrderView], known_client_order_ids: set) -> bool:
        """Return True if everything the broker shows matches something we
        recognize. If the broker has an open order whose client_order_id we
        never generated (deterministic pt-<symbol>-<side>-<date>), we cannot
        explain it -- HALT rather than guess whether it's safe to layer more
        orders on top of it."""
        for order in open_orders:
            if order.client_order_id not in known_client_order_ids:
                self._log(
                    "error",
                    f"reconcile: unrecognized open order {order.id} ({order.client_order_id}) "
                    f"for {order.symbol} -- halting",
                )
                return False
        return True

    def _known_client_order_ids(self) -> set:
        """All client_order_ids we could plausibly have generated: today's
        and yesterday's, for every whitelisted symbol and side, PLUS the
        one-time core-satellite bootstrap ids (see core.py). Yesterday is
        included so a tactical order submitted just before midnight UTC and
        still open the next tick isn't mistaken for a stranger."""
        from datetime import timedelta

        today = datetime.now(timezone.utc).date()
        yesterday = today - timedelta(days=1)
        ids = set()
        for d in (today, yesterday):
            for symbol in self.cfg.symbols:
                for side in ("buy", "sell"):
                    ids.add(f"pt-{symbol}-{side}-{d.isoformat()}")
        for symbol in self.cfg.symbols:
            ids.add(core_client_order_id(symbol))
        return ids

    # -- history cache ----------------------------------------------------------

    def _refresh_history_if_needed(self) -> None:
        today = datetime.now(timezone.utc).date()
        if self._history_cache_date == today and self._history_cache:
            return
        fresh: Dict[str, List[float]] = {}
        for symbol in self.cfg.symbols:
            try:
                fresh[symbol] = self.broker.recent_closes(symbol, self.cfg.history_lookback_days)
            except BrokerError as e:
                self._log("warn", f"history refresh failed for {symbol}: {e}")
        self._history_cache = fresh
        self._history_cache_date = today
        self._log("info", f"refreshed daily history cache for {list(fresh.keys())}")

    # -- main tick --------------------------------------------------------------

    def tick(self) -> None:
        try:
            self._tick_inner()
            self.circuit_breakers.record_success()
        except Exception:
            tb = traceback.format_exc()
            # Print to stderr FIRST, unconditionally -- this is the one
            # place that must never depend on anything else (the in-memory
            # event log, runtime_state.json) working correctly. NSSM
            # captures stderr to logs\PaperTiger-Engine.err.log, so this is
            # what a human (or a future automated log-scanner) actually
            # sees if _write_state() itself is what's broken.
            print(f"[engine] unhandled exception in tick:\n{tb}", file=sys.stderr, flush=True)
            self._log("error", f"unhandled exception in tick: {tb}")
            action = self.circuit_breakers.record_error()
            if action == BreakerAction.HALT:
                self.kill_switch.trigger(KillMode.HALT, "consecutive tick errors exceeded threshold")
                self._halted = True
                notify.send_notification(
                    self.cfg, "Engine HALTED: consecutive errors",
                    f"{self.circuit_breakers.state.consecutive_errors} consecutive tick errors -- "
                    f"halted, no new entries. Latest error:\n{tb[-1500:]}",
                )
            # Best-effort state write even on failure, so staleness is visible
            # rather than silent.
            try:
                self._write_state()
            except Exception:
                print(f"[engine] also failed to write state after the error above:\n{traceback.format_exc()}", file=sys.stderr, flush=True)

    def _tick_inner(self) -> None:
        # Recomputed fresh each tick -- otherwise a HALT set earlier (e.g.
        # before an operator cleared the kill file) would keep printing
        # "engine halted -- sleeping" in the log forever even after trading
        # resumed normally, since nothing else in this method resets it.
        self._halted = False

        # 1. Kill file check -- obeyed before anything else happens.
        if self.kill_switch.is_triggered():
            mode = self.kill_switch.mode()
            self._halted = True
            account = positions = open_orders = None
            try:
                account = self.broker.account()
                positions = self.broker.positions()
                open_orders = self.broker.open_orders()
            except BrokerError as e:
                self._log("error", f"kill-file active but could not fetch broker state: {e}")

            if mode == KillMode.FLATTEN:
                self._log("warn", "kill switch in FLATTEN mode -- liquidating to cash")
                try:
                    self.broker.flatten_everything()
                except BrokerError as e:
                    self._log("error", f"flatten_everything failed: {e}")
            else:
                self._log("info", "kill switch in HALT mode -- holding, no new entries")

            self._write_state(account, positions, open_orders, kill_mode=mode)
            return

        # 2. Pull truth from the broker. The broker's view always wins over
        #    any local assumption about what should be true.
        account = self.broker.account()
        positions = self.broker.positions()
        open_orders = self.broker.open_orders()

        # 3. Reconcile: an order we don't recognize means something placed a
        #    trade outside this engine's own logic (a bug, a stale process,
        #    manual intervention) -- halt and let a human look.
        if not self._reconcile(open_orders, self._known_client_order_ids()):
            self.kill_switch.trigger(KillMode.HALT, "unrecognized open order during reconcile")
            self._halted = True
            notify.send_notification(
                self.cfg, "Engine HALTED: unrecognized order",
                "The broker shows an open order this engine didn't generate itself -- halted "
                "until a human looks. This could be a bug, a stale process, or manual "
                "intervention outside the bot.",
            )
            self._write_state(account, positions, open_orders, kill_mode=KillMode.HALT)
            return

        # 4. Circuit breakers, evaluated on the broker's own equity number.
        today = datetime.now(timezone.utc).date()
        breaker_action = self.circuit_breakers.check_equity(account.equity, today)
        if breaker_action == BreakerAction.FLATTEN:
            self._log(
                "warn",
                f"circuit breaker tripped FLATTEN (daily P/L {self.circuit_breakers.daily_pl_pct(account.equity):.2%}, "
                f"drawdown {self.circuit_breakers.drawdown_pct(account.equity):.2%})",
            )
            self.kill_switch.trigger(KillMode.FLATTEN, "circuit breaker: daily loss or drawdown limit breached")
            self._halted = True
            notify.send_notification(
                self.cfg, "Engine FLATTENED: circuit breaker tripped",
                f"Daily P/L {self.circuit_breakers.daily_pl_pct(account.equity):.2%}, "
                f"drawdown {self.circuit_breakers.drawdown_pct(account.equity):.2%} -- "
                f"liquidating everything to cash.",
            )
            try:
                self.broker.flatten_everything()
            except BrokerError as e:
                self._log("error", f"flatten_everything failed: {e}")
            self._write_state(account, positions, open_orders, kill_mode=KillMode.FLATTEN)
            return

        # 5. Only look for new trades while the market is open. Log only on
        #    the open<->closed transition, not every tick, so an overnight/
        #    weekend closure doesn't spam the event log with ~200 identical
        #    lines (LOOP_INTERVAL_SEC ticks happen the whole time regardless).
        market_open = self.broker.market_open()
        if not market_open:
            if self._market_open_prev is not False:
                self._log("info", "market closed -- no new entries until it reopens")
            self._market_open_prev = False
            self._write_state(account, positions, open_orders)
            return
        if self._market_open_prev is not True:
            self._log("info", "market reopened")
        self._market_open_prev = True

        self._refresh_history_if_needed()

        quotes = {}
        for symbol in self.cfg.symbols:
            try:
                quotes[symbol] = self.broker.quote(symbol)
            except BrokerError as e:
                self._log("warn", f"quote fetch failed for {symbol}: {e}")

        # 5b. One-time core-satellite bootstrap (see core.py). A no-op on
        # every tick after every symbol's core position is established.
        for line in self.core.ensure_core_positions(self.broker, account, quotes, open_orders):
            self._log("info", line)
        core_holdings = self.core.load()

        history = {}
        for symbol in self.cfg.symbols:
            cached = self._history_cache.get(symbol)
            quote = quotes.get(symbol)
            if cached is not None and quote is not None:
                history[symbol] = cached + [quote.mid]

        intents: List[OrderIntent] = propose(account, positions, quotes, history, self.cfg, core_holdings)

        # 6 & 7. Validate then submit survivors, one at a time, with an
        # idempotency key. On any doubt about whether a submit "actually
        # happened," look it up by client_order_id instead of guessing.
        today = datetime.now(timezone.utc).date()
        for intent in intents:
            if self._is_same_day_round_trip(intent.symbol, intent.side, today):
                self._log(
                    "info",
                    f"skipping {intent.side} for {intent.symbol}: the opposite side already "
                    f"traded today -- refusing a same-day round trip",
                )
                continue
            result = self.pre_trade.validate(intent, account, positions, quotes, open_orders, core_holdings)
            if not result.ok:
                self._log("info", f"rejected intent for {intent.symbol} ({intent.side}): {result.reason}")
                continue
            self._submit_intent(intent)

        self._write_state(account, positions, open_orders, core_holdings=core_holdings)

    def _is_same_day_round_trip(self, symbol: str, side: str, today) -> bool:
        """True if the OPPOSITE side already has an order today for this
        symbol -- i.e. placing `intent` would open-and-close (or
        close-and-reopen) a position in the same symbol on the same
        calendar day.

        Why this matters even though this is a CASH account (not margin):
        the classic "3 day-trades per 5 days under $25k" Pattern Day
        Trader rule is a MARGIN-account rule and doesn't directly apply
        here -- but a cash account has its own version of the same
        problem: buying again using proceeds from a same-day sale that
        hasn't settled yet (T+1) is a "good-faith violation," and repeated
        violations get the account restricted to settled-cash-only trades
        for 90 days. Since this bot's signal is evaluated against a daily
        bar's worth of history but re-checked every tick using a LIVE
        intraday price, a big enough intraday swing could otherwise flip
        buy->sell (or vice versa) on the same symbol within one day. This
        check refuses that outright: at most one tactical action per
        symbol per day, which structurally rules out a same-day round
        trip regardless of how often the loop ticks.
        """
        opposite_side = "sell" if side == "buy" else "buy"
        opposite_client_id = f"pt-{symbol}-{opposite_side}-{today.isoformat()}"
        return self.broker.order_by_client_id(opposite_client_id) is not None

    def _submit_intent(self, intent: OrderIntent) -> None:
        # Ambiguity handling: if we're not sure a previous submit went
        # through (e.g. this exact intent was already tried today), check
        # for an existing order under the same client_order_id FIRST rather
        # than submitting blind and possibly doubling up.
        existing = self.broker.order_by_client_id(intent.client_order_id)
        if existing is not None:
            self._log(
                "info",
                f"intent for {intent.symbol} ({intent.side}) already has an order "
                f"({existing.id}, status={existing.status}) -- not resubmitting",
            )
            return

        try:
            if intent.side == "buy":
                order = self.broker.submit_limit_buy(
                    intent.symbol, intent.notional_usd, intent.limit_price, intent.client_order_id
                )
            else:
                order = self.broker.submit_limit_sell(
                    intent.symbol, intent.qty, intent.limit_price, intent.client_order_id
                )
            self._log(
                "info",
                f"submitted {intent.side} for {intent.symbol}: {intent.reason}",
                order_id=order.id,
                client_order_id=order.client_order_id,
            )
            trade_log.append_trade(
                self.cfg.trade_log_file_path, source="tactical", symbol=intent.symbol, side=intent.side,
                reason=intent.reason, qty=intent.qty, notional_usd=intent.notional_usd,
                limit_price=intent.limit_price, order_id=order.id, client_order_id=order.client_order_id,
            )
        except BrokerError as e:
            # Ambiguous outcome (e.g. request timed out but may have landed):
            # resolve via lookup rather than retrying immediately.
            self._log("error", f"submit failed for {intent.symbol} ({intent.side}): {e} -- checking for ambiguous fill")
            confirmed = self.broker.order_by_client_id(intent.client_order_id)
            if confirmed is not None:
                self._log("warn", f"order for {intent.symbol} actually landed despite submit error: {confirmed.id}")
            else:
                self._log("warn", f"order for {intent.symbol} did not land -- will reconsider next tick")

    # -- run loop ------------------------------------------------------------

    def run_forever(self) -> None:
        self._log("info", "engine starting")
        while True:
            self.tick()
            if self._halted:
                self._log("info", "engine halted -- sleeping, but not exiting (an operator must intervene)")
            time.sleep(self.cfg.loop_interval_sec)
