"""
sleeves.py -- splitting one brokerage account into independent strategy sleeves.

A sleeve is one strategy with its own pool of money and its own symbols:
the buy-and-hold core (core.py), plus one sleeve per signal kind listed in
STRATEGY_SLEEVES (sma_crossover / rsi_reversion / ml_classifier). They all
share a single Alpaca account, which only knows about one combined cash
balance and one position per symbol. Two rules make that sharing safe:

  1. DISJOINT SYMBOLS. No symbol ever belongs to two sleeves, and no sleeve
     ever holds a core symbol. So every position at the broker belongs to
     exactly one sleeve, decided by its ticker alone -- there is never a
     question of whose shares are whose, two sleeves can never send
     opposing orders for the same stock, and the engine's existing
     per-symbol safety rules (same-day round trip, duplicate order,
     position cap) keep meaning exactly what they meant before.
     load_sleeves() enforces this on every read: any symbol listed twice,
     or overlapping core, is dropped from every sleeve (fail closed).
  2. VIRTUAL CASH, REBUILT FROM THE BROKER. The account has one cash
     balance, so each sleeve's cash is reconstructed every tick from the
     broker's own order history since the experiment started
     (compute_sleeve_ledger): its pool, minus what its filled buys cost,
     minus what its still-open buys could cost, plus what its filled sells
     brought in. An order belongs to whichever sleeve owns its SYMBOL --
     exact, because of rule 1 -- so even orders the sleeve didn't place
     itself (a FLATTEN's "sell everything" orders) land in the right
     ledger. Sleeve orders also carry the sleeve's id in their
     client_order_id (pt-<sleeve>-<SYMBOL>-<side>-<date>), for the
     same-day round-trip check and for humans reading the order history.
     Nothing is tracked in a local file that could drift from what
     actually happened.

The engine hands each sleeve a scoped view of the account (its own cash,
its own equity, its own symbols) and then runs the SAME strategy.propose()
and safety.PreTradeCheck it always has -- so a sleeve can never spend
money from another sleeve's pool, even though the real account (a paper
account with ~$100k in it) could afford far more.

sleeves.json (written once by scripts/start_sleeve_experiment.py) records
which symbols each sleeve owns, each pool, the start time, and every
symbol's starting price -- the latter so each sleeve can be compared
against simply buying and holding its own symbols, which is the fairest
measure of whether a signal is adding anything (raw returns across
DIFFERENT symbols mostly measure which stocks happened to go up).

Nothing in this module talks to the broker or places orders.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# signal kind -> the short sleeve id used in client_order_ids and the UI.
SLEEVE_IDS: Dict[str, str] = {
    "sma_crossover": "sma",
    "rsi_reversion": "rsi",
    "ml_classifier": "ml",
}
CORE_SLEEVE_ID = "core"
SLEEVE_LABELS: Dict[str, str] = {
    CORE_SLEEVE_ID: "Buy & hold (core)",
    "sma": "SMA crossover",
    "rsi": "RSI reversion",
    "ml": "ML classifier",
}
RESET_CLIENT_ID_PREFIX = "pt-reset-"

# Statuses where an order may still fill further -- its unfilled remainder
# is reserved out of the sleeve's cash so the sleeve can't spend it twice.
OPEN_ORDER_STATUSES = frozenset({
    "new", "accepted", "pending_new", "partially_filled", "accepted_for_bidding",
    "pending_replace", "pending_cancel", "pending_review", "held", "calculated",
})

_QTY_EPSILON = 1e-9


def sleeve_order_prefix(sleeve_id: str) -> str:
    return f"pt-{sleeve_id}-"


def sleeve_client_order_id(sleeve_id: str, symbol: str, side: str, as_of: date) -> str:
    return f"{sleeve_order_prefix(sleeve_id)}{symbol}-{side}-{as_of.isoformat()}"


def reset_client_order_id(symbol: str, as_of: date) -> str:
    """The one-time "sell everything that isn't core" orders placed by
    scripts/start_sleeve_experiment.py."""
    return f"{RESET_CLIENT_ID_PREFIX}{symbol}-sell-{as_of.isoformat()}"


def core_topup_client_order_id(symbol: str, as_of: date) -> str:
    """The one-time buy that tops a core position up to its pool share."""
    return f"pt-core-{symbol}-topup-{as_of.isoformat()}"


# --------------------------------------------------------------------------
# Sleeve definitions (sleeves.json)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SleeveSpec:
    sleeve_id: str
    signal_kind: str
    pool_usd: float
    symbols: Tuple[str, ...]
    start_prices: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SleevesConfig:
    started_at: Optional[datetime]   # None = no experiment started yet (no sleeve symbols)
    frozen: bool
    core_pool_usd: float
    core_start_cash: Optional[float]
    sleeves: Tuple[SleeveSpec, ...]
    dropped_symbols: Tuple[str, ...] = ()  # removed by the disjointness check
    settings: Dict = field(default_factory=dict)  # experiment_settings() at the start

    @property
    def all_symbols(self) -> Tuple[str, ...]:
        return tuple(s for spec in self.sleeves for s in spec.symbols)

    def sleeve_for_symbol(self, symbol: str) -> Optional[SleeveSpec]:
        for spec in self.sleeves:
            if symbol in spec.symbols:
                return spec
        return None


def empty_sleeves(cfg) -> SleevesConfig:
    """Every configured sleeve, with its configured pool but no symbols --
    so nothing trades. What load_sleeves() fails closed to."""
    return SleevesConfig(
        started_at=None,
        frozen=False,
        core_pool_usd=cfg.core_pool_usd,
        core_start_cash=None,
        sleeves=tuple(
            SleeveSpec(SLEEVE_IDS[kind], kind, cfg.sleeve_pool_usd, ()) for kind in cfg.sleeve_kinds()
        ),
    )


def _positive_float(value) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if value > 0 else None


def load_sleeves(cfg) -> SleevesConfig:
    """Read sleeves.json fresh. Fails closed to empty_sleeves(cfg) on a
    missing/malformed file or a missing start time. Only sleeves whose
    signal kind is in cfg.sleeve_kinds() are returned, in that order."""
    try:
        raw = json.loads(Path(cfg.sleeves_file_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return empty_sleeves(cfg)
    if not isinstance(raw, dict) or not isinstance(raw.get("sleeves"), dict):
        return empty_sleeves(cfg)
    try:
        started_at = datetime.fromisoformat(str(raw.get("started_at")))
    except ValueError:
        return empty_sleeves(cfg)
    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=timezone.utc)

    core_raw = raw.get("core") if isinstance(raw.get("core"), dict) else {}
    core_pool = _positive_float(core_raw.get("pool_usd")) or cfg.core_pool_usd
    core_start_cash = core_raw.get("start_cash")
    if isinstance(core_start_cash, bool) or not isinstance(core_start_cash, (int, float)):
        core_start_cash = None

    parsed: List[SleeveSpec] = []
    for kind in cfg.sleeve_kinds():
        sid = SLEEVE_IDS[kind]
        entry = raw["sleeves"].get(sid)
        if not isinstance(entry, dict):
            parsed.append(SleeveSpec(sid, kind, cfg.sleeve_pool_usd, ()))
            continue
        symbols = entry.get("symbols") if isinstance(entry.get("symbols"), list) else []
        clean = tuple(dict.fromkeys(s.strip().upper() for s in symbols if isinstance(s, str) and s.strip()))
        prices_raw = entry.get("start_prices") if isinstance(entry.get("start_prices"), dict) else {}
        start_prices = {}
        for sym, price in prices_raw.items():
            p = _positive_float(price)
            if isinstance(sym, str) and p is not None:
                start_prices[sym.strip().upper()] = p
        pool = _positive_float(entry.get("pool_usd")) or cfg.sleeve_pool_usd
        parsed.append(SleeveSpec(sid, kind, pool, clean, start_prices))

    # Disjointness: a symbol in core or in more than one sleeve is ambiguous
    # -- nobody can trade it.
    counts: Dict[str, int] = {}
    for spec in parsed:
        for sym in spec.symbols:
            counts[sym] = counts.get(sym, 0) + 1
    core_symbols = set(cfg.symbols)
    dropped = sorted(s for s, n in counts.items() if n > 1 or s in core_symbols)
    if dropped:
        drop = set(dropped)
        parsed = [
            SleeveSpec(p.sleeve_id, p.signal_kind, p.pool_usd,
                       tuple(s for s in p.symbols if s not in drop), p.start_prices)
            for p in parsed
        ]

    return SleevesConfig(
        started_at=started_at,
        frozen=bool(raw.get("frozen", True)),
        core_pool_usd=core_pool,
        core_start_cash=float(core_start_cash) if core_start_cash is not None else None,
        sleeves=tuple(parsed),
        dropped_symbols=tuple(dropped),
        settings=raw.get("settings") if isinstance(raw.get("settings"), dict) else {},
    )


def write_sleeves_file(
    path: str,
    started_at: datetime,
    core_pool_usd: float,
    core_start_cash: float,
    core_symbols: Sequence[str],
    sleeves: Dict[str, Dict],
    frozen: bool = True,
    settings: Optional[Dict] = None,
) -> None:
    """Atomic (tmp + rename) write of sleeves.json. `sleeves` maps sleeve id
    -> {"signal_kind", "pool_usd", "symbols", "start_prices"}; `settings` is
    experiment_settings() at the start."""
    payload = {
        "_comment": "Written by scripts/start_sleeve_experiment.py. Each signal sleeve owns its "
                    "own symbols and cash pool inside the one account; see sleeves.py.",
        "started_at": started_at.isoformat(),
        "frozen": frozen,
        "settings": settings or {},
        "core": {"pool_usd": core_pool_usd, "start_cash": round(core_start_cash, 2),
                 "symbols": list(core_symbols)},
        "sleeves": sleeves,
    }
    target = Path(path)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(target)


# Every setting that changes how a sleeve behaves. Recorded in sleeves.json
# when an experiment starts, and compared every tick against what's running
# now -- so a mid-experiment change (say, a different risk profile in week
# 4) is flagged on the Compare tab instead of silently muddying the results.
EXPERIMENT_SETTING_FIELDS = (
    "trade_size_pct", "max_position_pct", "max_concentration_pct", "cash_buffer_pct",
    "daily_loss_limit_pct", "max_drawdown_pct", "max_open_positions", "breaker_action",
    "signal_fast", "signal_slow", "signal_period", "signal_oversold", "signal_overbought",
    "signal_ml_buy_threshold", "signal_ml_sell_threshold",
)


def experiment_settings(effective_cfg, risk_profile: Optional[str]) -> Dict:
    """The settings in effect right now (risk profile already folded in)."""
    settings = {f: getattr(effective_cfg, f) for f in EXPERIMENT_SETTING_FIELDS}
    settings["risk_profile"] = risk_profile or "none (.env values)"
    return settings


def settings_changes(recorded: Dict, current: Dict) -> List[Dict]:
    """[{"setting", "at_start", "now"}] for every recorded setting that
    differs now. Empty if nothing was recorded (an older sleeves.json)."""
    return [
        {"setting": k, "at_start": v, "now": current.get(k)}
        for k, v in recorded.items()
        if k in current and current.get(k) != v
    ]


def dollar_limits(cfg, pool_usd: float) -> Dict[str, float]:
    """The risk profile's pool-relative sizing (cfg.trade_size_pct etc.)
    turned into the dollar fields strategy.py and safety.PreTradeCheck
    work in, for one sleeve's pool."""
    return {
        "target_trade_usd": round(cfg.trade_size_pct * pool_usd, 2),
        "max_position_usd": round(cfg.max_position_pct * pool_usd, 2),
        "cash_buffer_usd": round(cfg.cash_buffer_pct * pool_usd, 2),
    }


def cfg_for_sleeve(cfg, sleeve_id: str):
    """A copy of cfg narrowed to one sleeve -- its symbols, its signal, its
    pool as starting capital, and no core carve-out -- for the offline tools
    (train_ml_signal.py / backtest.py / walkforward.py --sleeve <id>).
    Exits with a clear message if that sleeve has no symbols yet."""
    from dataclasses import replace

    spec = next((s for s in load_sleeves(cfg).sleeves if s.sleeve_id == sleeve_id), None)
    if spec is None or not spec.symbols:
        raise SystemExit(
            f"sleeve {sleeve_id!r} has no symbols in {cfg.sleeves_file_path} -- is it in STRATEGY_SLEEVES, "
            f"and has scripts/start_sleeve_experiment.py been run?"
        )
    return replace(cfg, symbols=spec.symbols, signal_kind=spec.signal_kind,
                   seed_usd=spec.pool_usd, core_allocation_pct=0.0)


# --------------------------------------------------------------------------
# Assigning symbols to sleeves
# --------------------------------------------------------------------------

def deal_symbols(
    ranked_symbols: Sequence[str],
    sectors: Dict[str, str],
    sleeve_ids: Sequence[str],
    exclude: Iterable[str] = (),
) -> Dict[str, List[str]]:
    """Split a liquidity-ranked symbol list across sleeves so each gets a
    similar mix. Symbols are grouped by sector (unknown -> "other"); within
    each sector they're dealt out back-and-forth (A,B,C,C,B,A,...), and each
    sector starts with whichever sleeve currently has the fewest symbols,
    so no sleeve always gets a sector's most liquid name and the totals
    stay within one of each other."""
    excluded = set(exclude)
    groups: Dict[str, List[str]] = {}
    for sym in dict.fromkeys(ranked_symbols):
        if sym in excluded:
            continue
        groups.setdefault(sectors.get(sym, "other"), []).append(sym)

    result: Dict[str, List[str]] = {sid: [] for sid in sleeve_ids}
    n = len(sleeve_ids)
    if n == 0:
        return result
    for g_index, syms in enumerate(groups.values()):
        order = sorted(sleeve_ids, key=lambda sid: (len(result[sid]), (sleeve_ids.index(sid) - g_index) % n))
        for i, sym in enumerate(syms):
            pos = i % n
            if (i // n) % 2 == 1:
                pos = n - 1 - pos
            result[order[pos]].append(sym)
    return result


# --------------------------------------------------------------------------
# Ledgers -- each sleeve's cash and value, rebuilt from broker orders
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SleeveLedger:
    sleeve_id: str
    signal_kind: str
    pool_usd: float
    cash: float              # spendable now (open buys already reserved out)
    reserved_usd: float      # remaining cost of still-open buy orders
    positions_value: float
    benchmark_equity: Optional[float]  # buy-and-hold of this sleeve's own symbols
    open_positions: int
    filled_orders: int
    round_trips: int
    winning_round_trips: int

    @property
    def equity(self) -> float:
        return self.cash + self.reserved_usd + self.positions_value

    @property
    def return_pct(self) -> float:
        return (self.equity / self.pool_usd - 1.0) if self.pool_usd > 0 else 0.0

    @property
    def benchmark_return_pct(self) -> Optional[float]:
        if self.benchmark_equity is None or self.pool_usd <= 0:
            return None
        return self.benchmark_equity / self.pool_usd - 1.0

    @property
    def excess_return_pct(self) -> Optional[float]:
        bench = self.benchmark_return_pct
        return None if bench is None else self.return_pct - bench

    @property
    def win_rate(self) -> Optional[float]:
        return self.winning_round_trips / self.round_trips if self.round_trips else None

    def to_dict(self) -> Dict:
        return {
            "sleeve_id": self.sleeve_id,
            "label": SLEEVE_LABELS.get(self.sleeve_id, self.sleeve_id),
            "signal_kind": self.signal_kind,
            "pool_usd": self.pool_usd,
            "cash": round(self.cash, 2),
            "reserved_usd": round(self.reserved_usd, 2),
            "positions_value": round(self.positions_value, 2),
            "equity": round(self.equity, 2),
            "return_pct": self.return_pct,
            "benchmark_equity": round(self.benchmark_equity, 2) if self.benchmark_equity is not None else None,
            "benchmark_return_pct": self.benchmark_return_pct,
            "excess_return_pct": self.excess_return_pct,
            "open_positions": self.open_positions,
            "filled_orders": self.filled_orders,
            "round_trips": self.round_trips,
            "winning_round_trips": self.winning_round_trips,
            "win_rate": self.win_rate,
        }


def benchmark_equity(spec: SleeveSpec, prices: Dict[str, float]) -> Optional[float]:
    """The pool split equally across the sleeve's symbols at their start
    prices, marked to `prices` (a symbol with no current price counts at
    its start value)."""
    priced = [s for s in spec.symbols if spec.start_prices.get(s)]
    if not priced:
        return None
    per_symbol = spec.pool_usd / len(priced)
    total = 0.0
    for sym in priced:
        start = spec.start_prices[sym]
        now = prices.get(sym, start)
        total += per_symbol / start * now
    return total


def compute_sleeve_ledger(
    spec: SleeveSpec,
    orders: Iterable,
    positions: Dict[str, object],
    prices: Dict[str, float],
) -> SleeveLedger:
    owned = set(spec.symbols)
    cash = spec.pool_usd
    reserved = 0.0
    filled_orders = 0
    round_trips = 0
    wins = 0
    cycles: Dict[str, Dict[str, float]] = {}  # symbol -> open qty/cost/proceeds of the current round trip

    # By symbol, not by order id: no symbol belongs to two sleeves, so this
    # is exact -- and it also catches orders the sleeve didn't place itself,
    # like a FLATTEN's untagged "sell everything" orders.
    ordered = sorted(
        (o for o in orders if o.symbol in owned),
        key=lambda o: o.submitted_at or datetime.min.replace(tzinfo=timezone.utc),
    )
    for o in ordered:
        filled_qty = o.filled_qty or 0.0
        filled_value = filled_qty * (o.filled_avg_price or 0.0)
        if o.side == "buy":
            cash -= filled_value
            if o.status in OPEN_ORDER_STATUSES:
                if o.qty is not None and o.limit_price:
                    reserved += max(0.0, o.qty - filled_qty) * o.limit_price
                elif o.notional is not None:
                    reserved += max(0.0, o.notional - filled_value)
        else:
            cash += filled_value

        if filled_qty <= _QTY_EPSILON:
            continue
        filled_orders += 1
        cycle = cycles.setdefault(o.symbol, {"qty": 0.0, "cost": 0.0, "proceeds": 0.0})
        if o.side == "buy":
            cycle["qty"] += filled_qty
            cycle["cost"] += filled_value
        else:
            cycle["qty"] -= filled_qty
            cycle["proceeds"] += filled_value
            if cycle["qty"] <= _QTY_EPSILON:
                round_trips += 1
                if cycle["proceeds"] > cycle["cost"]:
                    wins += 1
                cycles[o.symbol] = {"qty": 0.0, "cost": 0.0, "proceeds": 0.0}

    held = [positions[s] for s in spec.symbols if s in positions and positions[s].qty > _QTY_EPSILON]
    return SleeveLedger(
        sleeve_id=spec.sleeve_id,
        signal_kind=spec.signal_kind,
        pool_usd=spec.pool_usd,
        cash=cash - reserved,
        reserved_usd=reserved,
        positions_value=sum(p.market_value for p in held),
        benchmark_equity=benchmark_equity(spec, prices),
        open_positions=len(held),
        filled_orders=filled_orders,
        round_trips=round_trips,
        winning_round_trips=wins,
    )


def compute_core_ledger(
    core_symbols: Sequence[str],
    pool_usd: float,
    start_cash: Optional[float],
    positions: Dict[str, object],
    core_holdings: Dict[str, float],
    orders: Iterable = (),
) -> SleeveLedger:
    """Core's cash is what was left over at the start (or, before any
    experiment, its pool minus what its recorded shares cost), adjusted by
    any core-symbol order filled since the start -- normally none, since
    core never trades again, but a FLATTEN sells it and the proceeds must
    stay core's. Its value counts only the shares recorded as core that are
    actually still held. It is its own benchmark."""
    value = 0.0
    cost = 0.0
    held = 0
    for sym in core_symbols:
        qty = core_holdings.get(sym, 0.0)
        pos = positions.get(sym)
        if qty <= _QTY_EPSILON or pos is None or pos.qty <= _QTY_EPSILON:
            continue
        held += 1
        value += min(qty, pos.qty) * pos.current_price
        cost += min(qty, pos.qty) * pos.avg_entry_price
    cash = start_cash if start_cash is not None else pool_usd - cost
    core = set(core_symbols)
    for o in orders:
        if o.symbol in core and o.filled_qty:
            filled_value = o.filled_qty * (o.filled_avg_price or 0.0)
            cash += filled_value if o.side == "sell" else -filled_value
    return SleeveLedger(
        sleeve_id=CORE_SLEEVE_ID,
        signal_kind="buy_and_hold",
        pool_usd=pool_usd,
        cash=cash,
        reserved_usd=0.0,
        positions_value=value,
        benchmark_equity=None,
        open_positions=held,
        filled_orders=0,
        round_trips=0,
        winning_round_trips=0,
    )
