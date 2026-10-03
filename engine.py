"""
engine.py -- the live trading loop.

This is the one place where every other module gets wired together against
a real (paper, by default) broker connection. The loop is intentionally
linear and boring -- read the tick() method top to bottom and it tells you
the entire safety story:

    0. Fold in the live risk profile + the strategy sleeves (see
       _build_effective_cfg) -> everything below uses effective_cfg, never
       the raw self.cfg, for anything risk/symbol-related.
    1. Is the kill file present?            -> obey it, do nothing new.
    2. Pull truth from the broker.           -> broker is always authoritative.
    3. Reconcile against our own records.    -> a surprise order means HALT.
    4. Rebuild every sleeve's cash and value from the broker's order history,
       then run circuit breakers on the total. -> may escalate to FLATTEN.
    5. If the market's open, top up core (one-time) and, for EACH signal
       sleeve separately (_trade_sleeve), ask strategy to propose for that
       sleeve's own symbols only:
    6.   filter same-day round trips and the sleeve's max_open_positions cap,
    7.   validate every remaining proposal through PreTradeCheck against the
         sleeve's own cash -- never the whole account's,
    8.   submit only the survivors, with an idempotency key.
    9. Write a runtime_state.json snapshot, no matter what happened.

Strategy sleeves (see sleeves.py): core buy-and-hold plus one sleeve per
signal in STRATEGY_SLEEVES share this one account, each with its own pool
of money and its own symbols -- no symbol ever belongs to two sleeves.

Any unhandled exception anywhere in a tick is caught at the very top level
and treated as an error tick -- feeding the consecutive-error circuit
breaker, which eventually HALTs. We never let "something went wrong that we
didn't anticipate" turn into "so let's try again immediately."

One narrow, verifiable exception to "an operator must intervene": if every
error in that streak was the broker/API itself failing (BrokerError -- e.g.
Alpaca returning 5xx), never a logic bug, a reconcile mismatch, or a
circuit-breaker trip, the resulting HALT is tagged with
safety.BROKER_CONNECTIVITY_HALT_REASON. Step 1 below probes the broker once
per tick while that specific HALT is active and auto-clears it the moment a
call actually succeeds -- "is the broker responding" is a fact the engine
can check for itself, unlike the reason behind any other halt. Every other
HALT/FLATTEN reason is still only ever cleared by a human.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from collections import deque
from dataclasses import asdict, replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import equity_history
import notify
import sleeve_history
import trade_log
from broker import Broker, BrokerError, OrderView
from core import CoreAllocator, core_client_order_id
from sleeves import (
    CORE_SLEEVE_ID,
    SLEEVE_LABELS,
    SleeveLedger,
    SleeveSpec,
    compute_core_ledger,
    compute_sleeve_ledger,
    core_topup_client_order_id,
    dollar_limits,
    empty_sleeves,
    load_sleeves,
    reset_client_order_id,
    sleeve_client_order_id,
)
from safety import (
    BROKER_CONNECTIVITY_HALT_REASON,
    DEFAULT_RISK_PROFILE,
    RISK_PROFILE_TUNABLE_FIELDS,
    BreakerAction,
    CircuitBreakers,
    KillMode,
    KillSwitch,
    PreTradeCheck,
    RiskProfileStore,
)
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
        self.risk_profile_store = RiskProfileStore(cfg.risk_profile_file_path)

        # Recomputed fresh at the top of every _tick_inner() call (see
        # _build_effective_cfg) by folding the live risk profile + every
        # strategy sleeve's symbols on top of the static self.cfg. Seeded to
        # self.cfg here purely so it's never undefined if something reads it
        # before the first tick completes. self.cfg itself is NEVER touched
        # by this -- core.py's CoreAllocator stays pinned to it directly, so
        # core sizing can't be affected by profile or sleeve changes.
        self.effective_cfg = cfg
        self._current_risk_profile_name = DEFAULT_RISK_PROFILE

        # symbol -> cached completed daily closes (refreshed once per day)
        self._history_cache: Dict[str, List[float]] = {}
        self._history_cache_date: Optional[date] = None

        self._events: Deque[dict] = deque(maxlen=_EVENT_LOG_MAXLEN)
        self._halted = False  # recomputed every tick in _tick_inner(); reflects
        # *this tick's* outcome, not a permanent latch -- the kill-switch file
        # (checked independently each tick) is what actually gates trading.
        self._consecutive_broker_errors = 0  # in lockstep with
        # circuit_breakers.state.consecutive_errors except it resets to 0 on
        # any non-BrokerError exception, so at HALT time we can tell whether
        # EVERY error in the streak was the broker/API itself failing (see
        # BROKER_CONNECTIVITY_HALT_REASON) versus a mix that includes a real
        # bug -- only the former is ever eligible for auto-recovery.
        self._broker_halt_logged = False  # true once we've logged the first
        # "still failing" line for the current broker-connectivity HALT, so
        # the every-5-minutes retry probe doesn't spam the event log.
        self._market_open_prev: Optional[bool] = None  # None until the first
        # tick observes it -- lets us log only on open<->closed transitions
        # instead of every 5-minute tick overnight/weekends.
        self._cap_skip_logged: Dict[str, Tuple[date, frozenset]] = {}  # per
        # sleeve: the (day, symbol-set) of the last "skipping buy(s) at the
        # position cap" summary we logged -- at most one per day per distinct
        # set of skipped symbols, so a cap that's simply staying full all day
        # doesn't re-announce itself every 5 minutes. Logs again the moment
        # the actual set of skipped symbols changes, even same-day.

        # Strategy sleeves (see sleeves.py): re-read from sleeves.json every
        # tick in _build_effective_cfg, like the risk profile.
        self.sleeves = empty_sleeves(cfg)
        self._ledgers: Dict[str, SleeveLedger] = {}  # last computed, per sleeve id
        self._ledgers_fresh = False  # computed THIS tick (vs. carried over)
        self._managed_equity: Optional[float] = None  # sum of every sleeve's equity
        self._once_per_day_logged: Dict[str, date] = {}  # message key -> day last logged

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

    def _log_once_per_day(self, key: str, level: str, message: str) -> None:
        today = datetime.now(timezone.utc).date()
        if self._once_per_day_logged.get(key) != today:
            self._once_per_day_logged[key] = today
            self._log(level, message)

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
        # Breaker percentages are measured on the money the strategies
        # actually manage, not the whole account (see _tick_inner step 4).
        equity = self._managed_equity
        if account is not None:
            equity_history.append_snapshot(
                self.cfg.equity_history_file_path,
                equity=account.equity,
                cash=account.cash,
                positions_value={sym: p.market_value for sym, p in (positions or {}).items()},
            )
        if account is not None and self._ledgers_fresh and self._ledgers:
            sleeve_history.record_today(
                self.cfg.sleeve_history_file_path,
                {
                    sid: {
                        "equity": round(led.equity, 2),
                        "cash": round(led.cash + led.reserved_usd, 2),
                        "benchmark": round(led.benchmark_equity, 2) if led.benchmark_equity is not None else None,
                    }
                    for sid, led in self._ledgers.items()
                },
            )
        sleeve_symbols = {CORE_SLEEVE_ID: list(self.cfg.symbols)}
        sleeve_symbols.update({spec.sleeve_id: list(spec.symbols) for spec in self.sleeves.sleeves})
        sleeves_state = {}
        for sid, ledger in self._ledgers.items():
            entry = ledger.to_dict()
            entry["symbols"] = sleeve_symbols.get(sid, [])
            sleeves_state[sid] = entry
        state = {
            "written_at": datetime.now(timezone.utc).isoformat(),
            "halted": self._halted,
            "kill_mode": kill_mode.value if kill_mode else None,
            "account": asdict(account) if account is not None else None,
            "positions": {sym: asdict(p) for sym, p in (positions or {}).items()},
            "open_orders": [asdict(o) for o in (open_orders or [])],
            "core_holdings": core_holdings or self.core.load(),
            "managed_equity": round(equity, 2) if equity is not None else None,
            "daily_pl_pct": self.circuit_breakers.daily_pl_pct(equity) if equity is not None else None,
            "drawdown_pct": self.circuit_breakers.drawdown_pct(equity) if equity is not None else None,
            "circuit_breakers": self.circuit_breakers.to_dict(),
            # One entry per strategy sleeve (core first), each with its own
            # cash, value, buy-and-hold benchmark and trade stats -- what the
            # dashboard's Compare tab renders. See sleeves.py.
            "sleeves": sleeves_state,
            "experiment": {
                "started_at": self.sleeves.started_at.isoformat() if self.sleeves.started_at else None,
                "frozen": self.sleeves.frozen,
                "dropped_symbols": list(self.sleeves.dropped_symbols),
            },
            "events": list(self._events),
            # A read-only snapshot of the operationally-relevant config, for
            # the dashboard's Status tab -- no credentials, just what's
            # actually running right now (loop cadence, signal params, caps).
            "config_snapshot": {
                "symbols": list(self.cfg.symbols),  # core sleeve's symbols -- see core.py
                "sleeve_symbols": sleeve_symbols,
                "strategy_sleeves": list(self.cfg.sleeve_kinds()),
                "core_pool_usd": self.sleeves.core_pool_usd,
                "sleeve_pool_usd": self.cfg.sleeve_pool_usd,
                "alpaca_paper": self.cfg.alpaca_paper,
                "signal_fast": self.cfg.signal_fast,
                "signal_slow": self.cfg.signal_slow,
                "signal_period": self.cfg.signal_period,
                "signal_oversold": self.cfg.signal_oversold,
                "signal_overbought": self.cfg.signal_overbought,
                "trade_size_pct": self.effective_cfg.trade_size_pct,
                "max_position_pct": self.effective_cfg.max_position_pct,
                "cash_buffer_pct": self.effective_cfg.cash_buffer_pct,
                # The same three, in dollars, for one signal sleeve's pool.
                "per_sleeve_dollars": dollar_limits(self.effective_cfg, self.cfg.sleeve_pool_usd),
                "max_concentration_pct": self.effective_cfg.max_concentration_pct,
                "daily_loss_limit_pct": self.effective_cfg.daily_loss_limit_pct,
                "max_drawdown_pct": self.effective_cfg.max_drawdown_pct,
                "max_open_positions": self.effective_cfg.max_open_positions,
                "loop_interval_sec": self.cfg.loop_interval_sec,
                # What the dashboard's Config tab actually renders for the
                # tunable section -- the engine's own last-applied values,
                # not a re-derivation, so it can never drift from reality.
                "risk_profile": {
                    "name": self._current_risk_profile_name,
                    "effective": {f: getattr(self.effective_cfg, f) for f in RISK_PROFILE_TUNABLE_FIELDS},
                },
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

    def _known_client_order_ids(self, extra_symbols: Tuple[str, ...] = ()) -> set:
        """All client_order_ids we could plausibly have generated, for today
        and yesterday (so an order placed just before midnight UTC and still
        open the next tick isn't mistaken for a stranger):
          - each sleeve's own ids for its own symbols (pt-<sleeve>-...),
          - the one-time core ids (bootstrap buys, see core.py, and the
            top-ups / "sell everything" orders from
            scripts/start_sleeve_experiment.py),
          - the older un-prefixed pt-<SYMBOL>-<side>-<date> ids, so orders
            from before sleeves existed don't trip reconcile on the
            switch-over day.
        `extra_symbols` should be the broker's currently-held symbols, so a
        position outside every current list still isn't a stranger."""
        from datetime import timedelta

        today = datetime.now(timezone.utc).date()
        yesterday = today - timedelta(days=1)
        all_symbols = set(self.effective_cfg.symbols) | set(extra_symbols)
        ids = set()
        for d in (today, yesterday):
            for symbol in all_symbols:
                for side in ("buy", "sell"):
                    ids.add(f"pt-{symbol}-{side}-{d.isoformat()}")
                ids.add(reset_client_order_id(symbol, d))
            for spec in self.sleeves.sleeves:
                for symbol in spec.symbols:
                    for side in ("buy", "sell"):
                        ids.add(sleeve_client_order_id(spec.sleeve_id, symbol, side, d))
            for symbol in self.cfg.symbols:
                ids.add(core_topup_client_order_id(symbol, d))
        for symbol in self.cfg.symbols:
            ids.add(core_client_order_id(symbol))
        return ids

    # -- history cache ----------------------------------------------------------

    def _refresh_history_if_needed(self) -> None:
        today = datetime.now(timezone.utc).date()
        if self._history_cache_date == today and self._history_cache:
            return
        fresh: Dict[str, List[float]] = {}
        for symbol in self.effective_cfg.symbols:
            try:
                fresh[symbol] = self.broker.recent_closes(symbol, self.cfg.history_lookback_days)
            except BrokerError as e:
                self._log("warn", f"history refresh failed for {symbol}: {e}")
        self._history_cache = fresh
        self._history_cache_date = today
        self._log("info", f"refreshed daily history cache for {list(fresh.keys())}")

    # -- live risk profile + strategy sleeves ---------------------------------

    def _build_effective_cfg(self):
        """Fold the live-reloadable risk profile (safety.RiskProfileStore)
        and the strategy sleeves' symbols (sleeves.json, see sleeves.py) on
        top of the static, frozen self.cfg loaded at startup. Re-derived
        fresh every tick -- neither source is cached, mirroring the kill
        switch's own "read the file every tick" posture, and both fail
        closed on anything missing or malformed (no profile override; no
        sleeve symbols, so no signal trading).

        effective_cfg.symbols is core + every sleeve's symbols: the full set
        the engine fetches history and quotes for and recognizes orders
        for. Each sleeve still trades only its own symbols -- see
        _trade_sleeve, which narrows this per sleeve.

        self.cfg itself is never mutated (still frozen, still the single
        source of truth for locked fields); this always returns a NEW
        Config via dataclasses.replace(). core.py's CoreAllocator is
        deliberately constructed with -- and stays pinned to -- self.cfg
        directly, never this, so core sizing can never be affected by a
        profile switch or the sleeves file.
        """
        profile_state = self.risk_profile_store.load()
        # Display-only fallback -- resolve() below correctly applies NOTHING
        # when profile_state.profile is None, so effective_cfg still reflects
        # self.cfg's own .env-configured values, not this cosmetic default.
        self._current_risk_profile_name = profile_state.profile or DEFAULT_RISK_PROFILE
        self.sleeves = load_sleeves(self.cfg)
        if self.sleeves.dropped_symbols:
            self._log_once_per_day(
                "dropped-symbols", "warn",
                f"sleeves.json lists {', '.join(self.sleeves.dropped_symbols)} in more than one sleeve "
                f"or in core -- no sleeve will trade those until it's fixed",
            )
        effective_symbols = tuple(dict.fromkeys(list(self.cfg.symbols) + list(self.sleeves.all_symbols)))
        return replace(self.cfg, symbols=effective_symbols, **profile_state.resolve())

    # -- main tick --------------------------------------------------------------

    def tick(self) -> None:
        try:
            self._tick_inner()
            self.circuit_breakers.record_success()
            self._consecutive_broker_errors = 0
        except Exception as exc:
            tb = traceback.format_exc()
            # Print to stderr FIRST, unconditionally -- this is the one
            # place that must never depend on anything else (the in-memory
            # event log, runtime_state.json) working correctly. NSSM
            # captures stderr to logs\PaperTiger-Engine.err.log, so this is
            # what a human (or a future automated log-scanner) actually
            # sees if _write_state() itself is what's broken.
            print(f"[engine] unhandled exception in tick:\n{tb}", file=sys.stderr, flush=True)
            if isinstance(exc, BrokerError):
                self._consecutive_broker_errors += 1
            else:
                self._consecutive_broker_errors = 0
            action = self.circuit_breakers.record_error()
            # A broker/API failure is an expected, external kind of error --
            # one line in the event log is enough (the full traceback is
            # still in the stderr log above). Anything else is a potential
            # bug in our own code, so it keeps the full traceback.
            if isinstance(exc, BrokerError):
                self._log(
                    "warn",
                    f"broker/API error ({self.circuit_breakers.state.consecutive_errors} of "
                    f"{self.cfg.max_consecutive_errors} before HALT): {exc}",
                )
            else:
                self._log("error", f"unhandled exception in tick: {tb}")
            if action == BreakerAction.HALT:
                # Only tag this as auto-recoverable if EVERY error in the
                # streak was a BrokerError -- a single non-broker exception
                # anywhere in the streak permanently disqualifies it (see
                # _consecutive_broker_errors above).
                all_broker = self._consecutive_broker_errors == self.circuit_breakers.state.consecutive_errors
                reason = BROKER_CONNECTIVITY_HALT_REASON if all_broker else "consecutive tick errors exceeded threshold"
                self.kill_switch.trigger(KillMode.HALT, reason)
                self._halted = True
                self._broker_halt_logged = False
                notify.send_notification(
                    self.cfg, "Engine HALTED: consecutive errors",
                    f"{self.circuit_breakers.state.consecutive_errors} consecutive tick errors -- "
                    f"halted, no new entries. "
                    + ("All were broker/API errors -- will auto-resume once the broker responds normally again. "
                       if all_broker else "")
                    + f"Latest error:\n{tb[-1500:]}",
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
        self._ledgers_fresh = False

        # 0. Fold the live risk profile + the strategy sleeves on top of the
        #    static self.cfg (see _build_effective_cfg). Done before
        #    anything else so every step below -- including the kill-switch
        #    and reconcile paths' own _write_state() calls -- sees a
        #    consistent, freshly-computed self.effective_cfg.
        self.effective_cfg = self._build_effective_cfg()
        self.pre_trade.cfg = self.effective_cfg
        self.circuit_breakers.cfg = self.effective_cfg

        # 1. Kill file check -- obeyed before anything else happens, with one
        #    narrow, verifiable exception: a HALT this engine tagged
        #    BROKER_CONNECTIVITY_HALT_REASON (every error in the streak that
        #    caused it was the broker/API itself failing) is safe to probe
        #    each tick and auto-clear the moment a broker call actually
        #    succeeds -- see the module docstring and KillSwitch.clear().
        #    Every other reason (an operator's own halt, a reconcile
        #    mismatch, a circuit-breaker trip) is untouched by this branch.
        if self.kill_switch.is_triggered():
            mode = self.kill_switch.mode()
            reason = self.kill_switch.reason()
            auto_recovered = False

            if mode == KillMode.HALT and reason == BROKER_CONNECTIVITY_HALT_REASON:
                try:
                    self.broker.account()
                except BrokerError as e:
                    self._halted = True
                    if not self._broker_halt_logged:
                        self._log("info", f"halted on broker/API errors -- probing each tick, still failing: {e}")
                        self._broker_halt_logged = True
                    self._write_state(kill_mode=mode)
                    return
                else:
                    self.kill_switch.clear()
                    self.circuit_breakers.state.consecutive_errors = 0
                    self._consecutive_broker_errors = 0
                    self._broker_halt_logged = False
                    auto_recovered = True
                    self._log("info", "broker/API connectivity recovered -- auto-resuming")
                    notify.send_notification(
                        self.cfg, "Engine auto-resumed",
                        "Broker/API connectivity recovered after a consecutive-error HALT -- "
                        "resuming normal operation automatically.",
                    )

            if not auto_recovered:
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
            # else: fall through into the normal tick flow below (steps 2-9)
            # immediately, rather than waiting for the next scheduled tick.

        # 2. Pull truth from the broker. The broker's view always wins over
        #    any local assumption about what should be true. The order
        #    history (every sleeve's cash is rebuilt from it) is fetched
        #    BEFORE positions on purpose: if an order fills in between, it
        #    shows as still-open (its cost reserved) AND as a position, so a
        #    sleeve's value briefly reads high rather than low -- and a
        #    phantom dip is the one that could falsely trip a breaker.
        account = self.broker.account()
        orders = self.broker.orders_since(self.sleeves.started_at) if self.sleeves.started_at else []
        positions = self.broker.positions()
        open_orders = self.broker.open_orders()

        # 3. Reconcile: an order we don't recognize means something placed a
        #    trade outside this engine's own logic (a bug, a stale process,
        #    manual intervention) -- halt and let a human look.
        if not self._reconcile(open_orders, self._known_client_order_ids(extra_symbols=tuple(positions.keys()))):
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

        # 4. Every sleeve's cash and value, rebuilt from the broker's own
        #    order history (see sleeves.py), then the circuit breakers --
        #    measured on the money the strategies actually manage (the sum
        #    of every sleeve), not the whole account. In a paper account
        #    holding ~$100k, a 3% daily-loss limit on the account would be
        #    $3,000 -- more than the strategies even have -- so it could
        #    never trip.
        today = datetime.now(timezone.utc).date()
        self._refresh_history_if_needed()
        core_holdings = self.core.load()
        self._update_ledgers(orders, positions, core_holdings, quotes={})
        managed = self._managed_equity
        breaker_action = self.circuit_breakers.check_equity(managed, today)
        if breaker_action == BreakerAction.FLATTEN:
            self._log(
                "warn",
                f"circuit breaker tripped FLATTEN on strategy-managed equity ${managed:,.2f} "
                f"(daily P/L {self.circuit_breakers.daily_pl_pct(managed):.2%}, "
                f"drawdown {self.circuit_breakers.drawdown_pct(managed):.2%})",
            )
            self.kill_switch.trigger(KillMode.FLATTEN, "circuit breaker: daily loss or drawdown limit breached")
            self._halted = True
            notify.send_notification(
                self.cfg, "Engine FLATTENED: circuit breaker tripped",
                f"Strategy-managed equity ${managed:,.2f}: daily P/L "
                f"{self.circuit_breakers.daily_pl_pct(managed):.2%}, "
                f"drawdown {self.circuit_breakers.drawdown_pct(managed):.2%} -- "
                f"liquidating everything to cash.",
            )
            try:
                self.broker.flatten_everything()
            except BrokerError as e:
                self._log("error", f"flatten_everything failed: {e}")
            self._write_state(account, positions, open_orders, kill_mode=KillMode.FLATTEN, core_holdings=core_holdings)
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
            self._write_state(account, positions, open_orders, core_holdings=core_holdings)
            return
        if self._market_open_prev is not True:
            self._log("info", "market reopened")
        self._market_open_prev = True

        quotes = {}
        for symbol in self.effective_cfg.symbols:
            try:
                quotes[symbol] = self.broker.quote(symbol)
            except BrokerError as e:
                self._log("warn", f"quote fetch failed for {symbol}: {e}")

        # 5b. One-time core bootstrap (see core.py). A no-op on every tick
        # after every core symbol's position is established.
        for line in self.core.ensure_core_positions(self.broker, account, quotes, open_orders):
            self._log("info", line)
        core_holdings = self.core.load()
        self._update_ledgers(orders, positions, core_holdings, quotes)

        # 6-8. Each signal sleeve proposes, filters, validates and submits
        # for its own symbols only, against its own cash.
        for spec in self.sleeves.sleeves:
            self._trade_sleeve(spec, self._ledgers[spec.sleeve_id], account, positions, quotes, open_orders, today)

        self._write_state(account, positions, open_orders, core_holdings=core_holdings)

    def _update_ledgers(self, orders, positions, core_holdings, quotes) -> None:
        """Recompute every sleeve's ledger and the managed-equity total.
        Prices for the buy-and-hold benchmarks: last cached daily close,
        overridden by a live quote, overridden by the broker's own position
        price -- freshest wins."""
        prices: Dict[str, float] = {s: closes[-1] for s, closes in self._history_cache.items() if closes}
        prices.update({s: q.mid for s, q in quotes.items()})
        prices.update({s: p.current_price for s, p in positions.items() if p.current_price})
        ledgers = {
            CORE_SLEEVE_ID: compute_core_ledger(
                self.cfg.symbols, self.sleeves.core_pool_usd, self.sleeves.core_start_cash, positions, core_holdings,
            )
        }
        for spec in self.sleeves.sleeves:
            ledgers[spec.sleeve_id] = compute_sleeve_ledger(spec, orders, positions, prices)
        self._ledgers = ledgers
        self._ledgers_fresh = True
        self._managed_equity = sum(led.equity for led in ledgers.values())

    def _trade_sleeve(self, spec: SleeveSpec, ledger: SleeveLedger, account, positions, quotes, open_orders, today) -> None:
        """Run one signal sleeve: the unchanged strategy.propose() and
        safety.PreTradeCheck, but handed a view of the account scoped to
        this sleeve -- its own symbols, its own cash as buying power, its
        own value as equity -- so the buying-power check, the position and
        concentration caps, and the whitelist all apply per sleeve. A
        sleeve can never spend another sleeve's money, however much the
        real account holds."""
        if not spec.symbols:
            return
        label = SLEEVE_LABELS.get(spec.sleeve_id, spec.sleeve_id)
        s_cfg = replace(
            self.effective_cfg, symbols=spec.symbols, signal_kind=spec.signal_kind,
            **dollar_limits(self.effective_cfg, spec.pool_usd),
        )
        if spec.signal_kind == "ml_classifier" and not Path(s_cfg.signal_model_path).exists():
            # Without this, the missing model would raise inside propose()
            # and HALT every sleeve via the consecutive-error breaker.
            self._log_once_per_day(
                f"ml-model-missing-{spec.sleeve_id}", "warn",
                f"[{spec.sleeve_id}] {label}: no trained model at {s_cfg.signal_model_path} -- this sleeve "
                f"sits out until one exists (python train_ml_signal.py --sleeve {spec.sleeve_id})",
            )
            return

        budget = max(0.0, min(ledger.cash, account.buying_power))
        s_account = replace(account, cash=budget, buying_power=budget, equity=ledger.equity)
        s_positions = {s: p for s, p in positions.items() if s in spec.symbols}
        history = {}
        for symbol in spec.symbols:
            cached = self._history_cache.get(symbol)
            quote = quotes.get(symbol)
            if cached is not None and quote is not None:
                history[symbol] = cached + [quote.mid]

        intents: List[OrderIntent] = propose(
            s_account, s_positions, quotes, history, s_cfg, core_holdings={}, sleeve_id=spec.sleeve_id,
        )

        # Validate then submit survivors, one at a time, with an
        # idempotency key. On any doubt about whether a submit "actually
        # happened," look it up by client_order_id instead of guessing.
        open_symbols = {s for s, p in s_positions.items() if p.qty > 1e-9}
        approved_this_tick: set = set()
        cap_skipped_symbols: List[str] = []  # batched into one summary line
        # below instead of one log line per symbol, and throttled to once per
        # day per distinct set (see self._cap_skip_logged).
        self.pre_trade.cfg = s_cfg
        try:
            for intent in intents:
                if self._is_same_day_round_trip(intent.symbol, intent.side, today, spec.sleeve_id):
                    self._log(
                        "info",
                        f"[{spec.sleeve_id}] skipping {intent.side} for {intent.symbol}: the opposite side "
                        f"already traded today -- refusing a same-day round trip",
                    )
                    continue
                if intent.side == "buy" and self._would_exceed_open_positions(
                    intent.symbol, open_symbols, approved_this_tick, s_cfg.max_open_positions
                ):
                    cap_skipped_symbols.append(intent.symbol)
                    continue
                result = self.pre_trade.validate(intent, s_account, s_positions, quotes, open_orders, {})
                if not result.ok:
                    self._log("info", f"[{spec.sleeve_id}] rejected intent for {intent.symbol} ({intent.side}): {result.reason}")
                    continue
                if intent.side == "buy":
                    approved_this_tick.add(intent.symbol)
                    # Later buys this same tick must see the money as spent.
                    remaining = s_account.buying_power - intent.notional_usd
                    s_account = replace(s_account, cash=remaining, buying_power=remaining)
                self._submit_intent(intent, spec.sleeve_id)
        finally:
            self.pre_trade.cfg = self.effective_cfg

        if cap_skipped_symbols:
            cap_skip_key = (today, frozenset(cap_skipped_symbols))
            if cap_skip_key != self._cap_skip_logged.get(spec.sleeve_id):
                self._cap_skip_logged[spec.sleeve_id] = cap_skip_key
                self._log(
                    "info",
                    f"[{spec.sleeve_id}] skipping {len(cap_skipped_symbols)} buy(s) at the "
                    f"{s_cfg.max_open_positions}-position cap (already full): {', '.join(cap_skipped_symbols)}",
                )

    def _would_exceed_open_positions(
        self, symbol: str, open_tactical_symbols: set, approved_this_tick: set, cap: int
    ) -> bool:
        """True if buying `symbol` would open a NEW distinct position in
        this sleeve beyond `cap`. Adding to a symbol the sleeve already holds
        (or already approved earlier this same tick) never counts against
        the cap -- this bounds how many DIFFERENT symbols one sleeve can hold
        at once, not the number of buy orders."""
        if symbol in open_tactical_symbols or symbol in approved_this_tick:
            return False
        return len(open_tactical_symbols | approved_this_tick) >= cap

    def _is_same_day_round_trip(self, symbol: str, side: str, today, sleeve_id: str) -> bool:
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

        Symbols belong to exactly one sleeve, so checking that sleeve's own
        ids is enough. A buy also checks the one-time "sell everything"
        order from scripts/start_sleeve_experiment.py, so a symbol sold off
        on the experiment's first day isn't bought straight back that day.
        """
        opposite_side = "sell" if side == "buy" else "buy"
        candidates = [sleeve_client_order_id(sleeve_id, symbol, opposite_side, today)]
        if side == "buy":
            candidates.append(reset_client_order_id(symbol, today))
        return any(self.broker.order_by_client_id(cid) is not None for cid in candidates)

    def _submit_intent(self, intent: OrderIntent, sleeve_id: str) -> None:
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
                f"[{sleeve_id}] submitted {intent.side} for {intent.symbol}: {intent.reason}",
                order_id=order.id,
                client_order_id=order.client_order_id,
            )
            trade_log.append_trade(
                self.cfg.trade_log_file_path, source=sleeve_id, symbol=intent.symbol, side=intent.side,
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
                # Skip this heartbeat entirely while probing a broker-
                # connectivity HALT -- step 1 already logs "still failing"
                # once per failure, and "an operator must intervene" would
                # be actively wrong here (no one needs to; it's retrying
                # itself). Every other halt reason keeps logging every tick,
                # unchanged from before.
                if self.kill_switch.reason() != BROKER_CONNECTIVITY_HALT_REASON:
                    self._log("info", "engine halted -- sleeping, but not exiting (an operator must intervene)")
            time.sleep(self.cfg.loop_interval_sec)
