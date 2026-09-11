"""
safety.py -- the SOFTWARE safety layer (capital preservation only).

Read this comment before touching anything else in this file: there are two
independent safety layers in this project, and it's important not to
confuse them.

  1. STRUCTURAL (load-bearing, lives in the Alpaca account itself): a CASH
     account, long-only. That is what makes "you can lose more than your
     balance" impossible. No code in this repository can override it --
     preflight.py checks for it, but the guarantee comes from the account
     type at the broker, not from anything we write here.
  2. SOFTWARE (this file): kill switch, circuit breakers, and pre-trade
     validation. These exist to protect capital WITHIN whatever balance you
     have. They are defense in depth, not the reason you can't lose more
     than you funded -- and they are only as good as the code that calls
     them (engine.py), so they fail closed: on any doubt, do nothing.

Nothing in this file makes network calls. KillSwitch reads/writes a small
local file; everything else is pure computation over the account/position/
quote/order snapshots the engine hands in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from strategy import OrderIntent


# --------------------------------------------------------------------------
# Kill switch
# --------------------------------------------------------------------------

# The exact reason string engine.py writes when it HALTs itself because every
# error in the consecutive-error streak was a BrokerError (the broker/API
# itself failing -- e.g. Alpaca returning 5xx), never a logic bug, a
# reconcile mismatch, or a circuit-breaker equity trip. engine.py checks for
# this exact string as the sole trigger for its one auto-clear exception
# (see KillSwitch.clear()) -- a HALT written with any other reason, including
# an operator's own note, is never auto-cleared.
BROKER_CONNECTIVITY_HALT_REASON = "consecutive tick errors exceeded threshold (broker/API connectivity)"


class KillMode(Enum):
    """HALT: stop opening new positions, hold what we have, keep reporting.
    FLATTEN: cancel all open orders and market-sell every position back to
    cash. FLATTEN is the stronger response -- it's what circuit breakers
    reach for on a daily-loss or drawdown breach, since at that point the
    priority is "stop losing money," not "preserve the current strategy."
    """

    HALT = "HALT"
    FLATTEN = "FLATTEN"


class KillSwitch:
    """A dead-simple file-based trigger.

    Presence of the file at `path` means "stop." The file's (stripped,
    upper-cased) contents pick the mode: literal "FLATTEN" selects FLATTEN,
    anything else (including an empty file) defaults to the safer HALT.
    Both the engine and the separate watchdog process only ever need to
    CREATE this file to stop trading -- neither needs special permissions
    or IPC, which keeps the watchdog trivially simple and hard to get wrong.
    """

    def __init__(self, path: str):
        self.path = Path(path)

    def is_triggered(self) -> bool:
        return self.path.exists()

    def mode(self) -> Optional[KillMode]:
        if not self.path.exists():
            return None
        try:
            contents = self.path.read_text(encoding="utf-8")
        except OSError:
            # Unreadable file still means "stop" -- fail closed, default HALT.
            return KillMode.HALT
        first_line = contents.strip().splitlines()[0].strip().upper() if contents.strip() else ""
        if first_line == "FLATTEN":
            return KillMode.FLATTEN
        return KillMode.HALT

    def reason(self) -> Optional[str]:
        """The second line of the kill file, if any -- the free-text reason
        passed to trigger(). None if the file is absent, unreadable, or has
        no second line."""
        if not self.path.exists():
            return None
        try:
            contents = self.path.read_text(encoding="utf-8")
        except OSError:
            return None
        lines = contents.strip("\n").splitlines()
        return lines[1].strip() if len(lines) > 1 else None

    def trigger(self, mode: KillMode, reason: str) -> None:
        """Create (or overwrite) the kill file. Idempotent -- calling this
        repeatedly with the same mode is harmless."""
        self.path.write_text(f"{mode.value}\n{reason}\n", encoding="utf-8")

    def clear(self) -> None:
        """Remove the kill file. This is a manual, operator-driven action
        (e.g. `rm HALT`) -- nothing in engine.py calls this automatically,
        with exactly one narrow exception: a HALT engine.py wrote itself with
        reason BROKER_CONNECTIVITY_HALT_REASON (every error in the streak was
        the broker/API failing, not a logic bug or an operator's own halt)
        gets auto-cleared once a broker call actually succeeds again, since
        "is the broker responding" is an externally verifiable fact rather
        than a guess that it's safe to keep going. Every other kill file --
        including a HALT with any other reason -- is never touched by
        anything but this method, called by a human."""
        self.path.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# Risk profile (live-reloadable, dashboard-writable subset of Config)
# --------------------------------------------------------------------------

# Fields a risk profile preset or manual override is ever allowed to touch.
# Deliberately narrow: risk-appetite dials only -- never account type, never
# the symbol whitelist, never strategy identity, never the same-day
# round-trip check (which has no backing Config field at all, so nothing
# here could reach it even if it tried). core_allocation_pct is excluded on
# purpose too: core.py treats it as a one-time bootstrap parameter tied to
# len(cfg.symbols), not something safe to resize while the process runs.
RISK_PROFILE_TUNABLE_FIELDS = (
    "target_trade_usd",
    "max_position_usd",
    "max_concentration_pct",
    "cash_buffer_usd",
    "daily_loss_limit_pct",
    "max_drawdown_pct",
    "max_open_positions",
)

RISK_PROFILE_PRESETS: Dict[str, Dict[str, float]] = {
    "conservative": {
        "target_trade_usd": 15.0,
        "max_position_usd": 40.0,
        "max_concentration_pct": 0.25,
        "cash_buffer_usd": 20.0,
        "daily_loss_limit_pct": 0.02,
        "max_drawdown_pct": 0.10,
        "max_open_positions": 3,
    },
    "normal": {
        "target_trade_usd": 25.0,
        "max_position_usd": 60.0,
        "max_concentration_pct": 0.35,
        "cash_buffer_usd": 10.0,
        "daily_loss_limit_pct": 0.03,
        "max_drawdown_pct": 0.15,
        "max_open_positions": 6,
    },
    "aggressive": {
        "target_trade_usd": 40.0,
        "max_position_usd": 90.0,
        "max_concentration_pct": 0.50,
        "cash_buffer_usd": 5.0,
        "daily_loss_limit_pct": 0.05,
        "max_drawdown_pct": 0.25,
        "max_open_positions": 10,
    },
}

DEFAULT_RISK_PROFILE = "normal"  # matches config.py's own field defaults -- display default only, see below


@dataclass(frozen=True)
class RiskProfileState:
    """profile=None means "no risk-profile file yet, or it's unreadable" --
    i.e. nobody has ever touched this dashboard control. resolve() returns
    {} in that case, deliberately: self.cfg (whatever the operator actually
    put in .env) must never be silently overwritten with a hardcoded
    preset's numbers just because the profile file happens to be missing or
    corrupt. A preset is ONLY ever applied once a human has explicitly
    picked one (or an override) via the dashboard, which is what makes the
    file exist with a valid profile name in the first place."""

    profile: Optional[str]
    overrides: Dict[str, float] = field(default_factory=dict)

    def resolve(self) -> Dict[str, float]:
        """Preset values merged with manual overrides -- overrides win.
        Empty (no-op) if no profile has ever been explicitly selected."""
        if self.profile is None:
            return {}
        resolved = dict(RISK_PROFILE_PRESETS[self.profile])
        resolved.update(self.overrides)
        return resolved


class RiskProfileStore:
    """Live-reloadable conservative/normal/aggressive profile plus manual
    per-field overrides, mirroring KillSwitch's file-based pattern: a bare
    Path, read fresh every tick, fail closed to "apply nothing" (see
    RiskProfileState.profile=None above) on anything missing, unreadable,
    or naming an unrecognized profile.

    Unlike the kill file, this one is meant to be written by a human via the
    dashboard, not just by the engine -- so the READER is the actual
    enforcement point: any key outside RISK_PROFILE_TUNABLE_FIELDS, any
    unrecognized profile name, or any non-numeric value is silently dropped
    rather than applied. That makes this file structurally incapable of
    ever touching a locked field, even from a hand-edited or corrupted copy.
    """

    def __init__(self, path: str):
        self.path = Path(path)

    def load(self) -> RiskProfileState:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            return RiskProfileState(profile=None)
        if not isinstance(raw, dict):
            return RiskProfileState(profile=None)

        profile = raw.get("profile")
        if profile not in RISK_PROFILE_PRESETS:
            return RiskProfileState(profile=None)

        overrides: Dict[str, float] = {}
        raw_overrides = raw.get("overrides")
        if isinstance(raw_overrides, dict):
            for key, value in raw_overrides.items():
                if key not in RISK_PROFILE_TUNABLE_FIELDS:
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                overrides[key] = value

        return RiskProfileState(profile=profile, overrides=overrides)

    def write(self, profile: str, overrides: Optional[Dict[str, float]] = None) -> None:
        """Atomic write (tmp + rename) -- a human may edit this via the
        dashboard while the engine reads it every tick, so a torn read of a
        half-written file should never be possible."""
        if profile not in RISK_PROFILE_PRESETS:
            raise ValueError(f"unknown risk profile {profile!r}")
        clean_overrides = {
            k: v for k, v in (overrides or {}).items() if k in RISK_PROFILE_TUNABLE_FIELDS
        }
        payload = {"profile": profile, "overrides": clean_overrides}
        tmp_path = self.path.with_name(self.path.name + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_path.replace(self.path)


# --------------------------------------------------------------------------
# Circuit breakers
# --------------------------------------------------------------------------

class BreakerAction(Enum):
    NONE = "none"
    HALT = "halt"
    FLATTEN = "flatten"


@dataclass
class CircuitBreakerState:
    """Mutable, in-process state for the circuit breakers. This resets on
    process restart -- that's fine, because engine.py's own rule is
    "unhandled exception anywhere -> HALT and hold," so a crash already
    stops new trading before this state would matter again. peak_equity and
    day_start_equity re-seed themselves from the first tick's account
    snapshot after a restart."""

    peak_equity: Optional[float] = None
    day_start_equity: Optional[float] = None
    day_start_date: Optional[date] = None
    consecutive_errors: int = 0


class CircuitBreakers:
    """Three independent breakers, evaluated every tick:

      - daily loss:    equity has fallen more than daily_loss_limit_pct
                        below the equity recorded at the start of today
                        -> FLATTEN (stop the bleeding, go to cash).
      - drawdown:       equity has fallen more than max_drawdown_pct below
                        its all-time-observed peak -> FLATTEN.
      - consecutive
        errors:         the engine loop has failed N times in a row
                        (network errors, broker errors, etc.) -> HALT
                        (we don't know enough to safely flatten, but we
                        stop opening anything new).

    `check_equity` is the routine per-tick check. `record_error` /
    `record_success` bracket the engine's per-tick try/except.
    """

    def __init__(self, cfg, state: Optional[CircuitBreakerState] = None):
        self.cfg = cfg
        self.state = state or CircuitBreakerState()

    def record_error(self) -> BreakerAction:
        self.state.consecutive_errors += 1
        if self.state.consecutive_errors >= self.cfg.max_consecutive_errors:
            return BreakerAction.HALT
        return BreakerAction.NONE

    def record_success(self) -> None:
        self.state.consecutive_errors = 0

    def check_equity(self, equity: float, as_of: date) -> BreakerAction:
        # (Re)seed peak equity.
        if self.state.peak_equity is None or equity > self.state.peak_equity:
            self.state.peak_equity = equity

        # (Re)seed the start-of-day equity whenever the calendar date rolls.
        if self.state.day_start_date != as_of or self.state.day_start_equity is None:
            self.state.day_start_date = as_of
            self.state.day_start_equity = equity

        worst = BreakerAction.NONE

        if self.state.day_start_equity and self.state.day_start_equity > 0:
            daily_loss_pct = (self.state.day_start_equity - equity) / self.state.day_start_equity
            if daily_loss_pct >= self.cfg.daily_loss_limit_pct:
                worst = BreakerAction.FLATTEN

        if self.state.peak_equity and self.state.peak_equity > 0:
            drawdown_pct = (self.state.peak_equity - equity) / self.state.peak_equity
            if drawdown_pct >= self.cfg.max_drawdown_pct:
                worst = BreakerAction.FLATTEN  # FLATTEN outranks nothing else here, but keep explicit

        return worst

    def daily_pl_pct(self, equity: float) -> float:
        if not self.state.day_start_equity:
            return 0.0
        return (equity - self.state.day_start_equity) / self.state.day_start_equity

    def drawdown_pct(self, equity: float) -> float:
        if not self.state.peak_equity:
            return 0.0
        return (self.state.peak_equity - equity) / self.state.peak_equity

    def to_dict(self) -> dict:
        return {
            "peak_equity": self.state.peak_equity,
            "day_start_equity": self.state.day_start_equity,
            "day_start_date": self.state.day_start_date.isoformat() if self.state.day_start_date else None,
            "consecutive_errors": self.state.consecutive_errors,
        }


# --------------------------------------------------------------------------
# Pre-trade validation
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CheckResult:
    ok: bool
    reason: str = ""


class PreTradeCheck:
    """Validates ONE OrderIntent against the current account/position/quote/
    order-book snapshot. Checks run in a fixed order and the FIRST failure
    rejects the intent -- we never partially apply an order, and we never
    let a later check "override" an earlier rejection. Nothing here talks
    to the broker; a rejection here means the intent never leaves engine.py.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def validate(
        self,
        intent: OrderIntent,
        account,                          # broker.AccountSnapshot
        positions: Dict[str, object],      # symbol -> broker.Position
        quotes: Dict[str, object],          # symbol -> broker.Quote
        open_orders: List[object],          # list[broker.OrderView]
        core_holdings: Optional[Dict[str, float]] = None,  # symbol -> core-reserved qty, see core.py
    ) -> CheckResult:
        core_holdings = core_holdings or {}
        checks = (
            self._check_whitelisted,
            self._check_long_only,
            self._check_fresh_quote,
            self._check_sane_qty_price,
            self._check_min_notional,
            self._check_limit_near_mid,
            self._check_sell_not_exceed_holdings,
            self._check_sell_preserves_core,
            self._check_buying_power,
            self._check_position_cap,
            self._check_concentration_cap,
            self._check_no_duplicate_order,
        )
        for check in checks:
            result = check(intent, account, positions, quotes, open_orders, core_holdings)
            if not result.ok:
                return result
        return CheckResult(ok=True, reason="all checks passed")

    # -- individual checks, in the order they run --

    def _check_whitelisted(self, intent, account, positions, quotes, open_orders, core_holdings) -> CheckResult:
        if intent.symbol not in self.cfg.symbols:
            return CheckResult(False, f"{intent.symbol} is not in the symbol whitelist")
        return CheckResult(True)

    def _check_long_only(self, intent, account, positions, quotes, open_orders, core_holdings) -> CheckResult:
        # This codebase never constructs a "short" side, but we check
        # explicitly anyway -- defense in depth against a future bug that
        # somehow produces one.
        if intent.side not in ("buy", "sell"):
            return CheckResult(False, f"unrecognized order side {intent.side!r} (long-only bot)")
        return CheckResult(True)

    def _check_fresh_quote(self, intent, account, positions, quotes, open_orders, core_holdings) -> CheckResult:
        quote = quotes.get(intent.symbol)
        if quote is None:
            return CheckResult(False, f"no quote available for {intent.symbol}")
        if quote.age_sec > self.cfg.quote_max_age_sec:
            return CheckResult(False, f"quote for {intent.symbol} is stale ({quote.age_sec:.1f}s old)")
        return CheckResult(True)

    def _check_sane_qty_price(self, intent, account, positions, quotes, open_orders, core_holdings) -> CheckResult:
        if intent.limit_price is None or intent.limit_price <= 0:
            return CheckResult(False, f"non-positive limit price for {intent.symbol}: {intent.limit_price}")
        if intent.side == "buy":
            if intent.notional_usd is None or intent.notional_usd <= 0:
                return CheckResult(False, f"non-positive notional for buy on {intent.symbol}")
        else:
            if intent.qty is None or intent.qty <= 0:
                return CheckResult(False, f"non-positive qty for sell on {intent.symbol}")
        return CheckResult(True)

    def _check_min_notional(self, intent, account, positions, quotes, open_orders, core_holdings) -> CheckResult:
        notional = intent.notional_usd if intent.side == "buy" else intent.qty * intent.limit_price
        if notional < self.cfg.min_notional_usd:
            return CheckResult(
                False,
                f"order notional ${notional:.2f} for {intent.symbol} is below min ${self.cfg.min_notional_usd:.2f}",
            )
        return CheckResult(True)

    def _check_limit_near_mid(self, intent, account, positions, quotes, open_orders, core_holdings) -> CheckResult:
        quote = quotes.get(intent.symbol)
        if quote is None:
            return CheckResult(False, f"no quote to sanity-check limit price for {intent.symbol}")
        mid = quote.mid
        if mid <= 0:
            return CheckResult(False, f"non-positive mid price for {intent.symbol}")
        distance_pct = abs(intent.limit_price - mid) / mid
        if distance_pct > self.cfg.limit_offset_pct:
            return CheckResult(
                False,
                f"limit price {intent.limit_price:.2f} for {intent.symbol} is "
                f"{distance_pct:.1%} from mid {mid:.2f} (fat-finger guard is {self.cfg.limit_offset_pct:.1%})",
            )
        return CheckResult(True)

    def _check_sell_not_exceed_holdings(self, intent, account, positions, quotes, open_orders, core_holdings) -> CheckResult:
        if intent.side != "sell":
            return CheckResult(True)
        held = positions.get(intent.symbol)
        held_qty = held.qty if held is not None else 0.0
        if intent.qty > held_qty + 1e-9:
            return CheckResult(
                False,
                f"refusing to sell {intent.qty} of {intent.symbol}, only hold {held_qty} (no accidental shorting)",
            )
        return CheckResult(True)

    def _check_sell_preserves_core(self, intent, account, positions, quotes, open_orders, core_holdings) -> CheckResult:
        """Independent enforcement of the core-satellite carve-out (see
        core.py): even if strategy.py somehow proposed selling into a core
        position -- a bug, not something it's supposed to do -- this check
        is the authoritative backstop that refuses it. Safety.py should
        never simply trust that the caller did its own arithmetic right."""
        if intent.side != "sell":
            return CheckResult(True)
        held = positions.get(intent.symbol)
        held_qty = held.qty if held is not None else 0.0
        core_qty = core_holdings.get(intent.symbol, 0.0)
        tactical_available = max(0.0, held_qty - core_qty)
        if intent.qty > tactical_available + 1e-9:
            return CheckResult(
                False,
                f"refusing to sell {intent.qty} of {intent.symbol}: only {tactical_available} is "
                f"tactical (non-core) qty -- {core_qty} is permanently reserved as core",
            )
        return CheckResult(True)

    def _check_buying_power(self, intent, account, positions, quotes, open_orders, core_holdings) -> CheckResult:
        if intent.side != "buy":
            return CheckResult(True)
        available = account.buying_power - self.cfg.cash_buffer_usd
        if intent.notional_usd > available:
            return CheckResult(
                False,
                f"buying ${intent.notional_usd:.2f} of {intent.symbol} would breach the "
                f"${self.cfg.cash_buffer_usd:.2f} cash buffer (only ${available:.2f} available)",
            )
        return CheckResult(True)

    def _check_position_cap(self, intent, account, positions, quotes, open_orders, core_holdings) -> CheckResult:
        if intent.side != "buy":
            return CheckResult(True)
        held = positions.get(intent.symbol)
        existing_value = held.market_value if held is not None else 0.0
        projected = existing_value + intent.notional_usd
        if projected > self.cfg.max_position_usd:
            return CheckResult(
                False,
                f"buying ${intent.notional_usd:.2f} more of {intent.symbol} would bring the "
                f"position to ${projected:.2f}, over the ${self.cfg.max_position_usd:.2f} cap",
            )
        return CheckResult(True)

    def _check_concentration_cap(self, intent, account, positions, quotes, open_orders, core_holdings) -> CheckResult:
        if intent.side != "buy":
            return CheckResult(True)
        if account.equity <= 0:
            return CheckResult(False, "account equity is non-positive; refusing to size any new buy")
        held = positions.get(intent.symbol)
        existing_value = held.market_value if held is not None else 0.0
        projected = existing_value + intent.notional_usd
        concentration = projected / account.equity
        if concentration > self.cfg.max_concentration_pct:
            return CheckResult(
                False,
                f"{intent.symbol} would be {concentration:.1%} of equity, over the "
                f"{self.cfg.max_concentration_pct:.1%} concentration cap",
            )
        return CheckResult(True)

    def _check_no_duplicate_order(self, intent, account, positions, quotes, open_orders, core_holdings) -> CheckResult:
        for order in open_orders:
            if order.symbol == intent.symbol:
                return CheckResult(
                    False,
                    f"an open order already exists for {intent.symbol} (id={getattr(order, 'id', '?')})",
                )
        return CheckResult(True)
