"""
start_sleeve_experiment.py -- one-time start of a strategy-sleeve comparison.

Sets the account up so the core buy-and-hold sleeve and every signal sleeve
in STRATEGY_SLEEVES start from the same line on the same day (see sleeves.py
for how sleeves share one account):

  1. Sell everything that isn't core: every position outside the core
     symbols, plus any shares in a core symbol beyond what core_holdings.json
     records as core (orders tagged pt-reset-..., limit sells just under
     the bid). Waits for the fills.
  2. Top core up so each core symbol is worth about CORE_POOL_USD /
     number of core symbols (orders tagged pt-core-<SYM>-topup-<date>), and
     record the extra shares in core_holdings.json. Core is never sold.
  3. Deal the liquidity-ranked tactical_universe.json out to the signal
     sleeves, sector by sector (sleeves.deal_symbols), record every
     symbol's starting price, and write sleeves.json. Core starts with
     exactly CORE_POOL_USD of value (its leftover cash is recorded), and
     every signal sleeve starts with SLEEVE_POOL_USD of cash -- so all of
     them start equal.

Safety:
  - Paper accounts only. It refuses to run against a live account.
  - DRY RUN BY DEFAULT: prints exactly what it would do and changes
    nothing. Pass --execute to actually trade (market must be open).
  - Stop the engine first; --execute refuses if runtime_state.json looks
    fresh (the engine still running), or if any order is still open.
  - Safe to re-run: orders are keyed by day, so a re-run waits on the same
    orders instead of placing duplicates. sleeves.json is only written
    once every sell and top-up has filled.

Usage:
    python scripts/refresh_tactical_universe.py      # make sure the ranking is fresh
    python scripts/start_sleeve_experiment.py        # dry run
    python scripts/start_sleeve_experiment.py --execute
    python scripts/start_sleeve_experiment.py --execute --restart   # replace a running experiment
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from broker import Broker, BrokerError  # noqa: E402
from config import load_config  # noqa: E402
from core import CoreAllocator  # noqa: E402
from refresh_tactical_universe import load_candidate_sectors  # noqa: E402
from sleeves import (  # noqa: E402
    OPEN_ORDER_STATUSES,
    SLEEVE_IDS,
    SLEEVE_LABELS,
    core_topup_client_order_id,
    deal_symbols,
    load_sleeves,
    reset_client_order_id,
    write_sleeves_file,
)

_QTY_EPSILON = 1e-6
_NUDGE = 0.001  # same 10 bps off the touch strategy.py uses


def plan_reset_sells(
    positions: Dict[str, object], core_symbols: Sequence[str], core_holdings: Dict[str, float]
) -> Tuple[List[Tuple[str, float]], List[str]]:
    """(symbol, qty) to sell so only recorded core shares remain, plus any
    warnings. A core symbol with no recorded core qty is left alone (we
    can't tell what part of it is core)."""
    sells: List[Tuple[str, float]] = []
    warnings: List[str] = []
    for sym, pos in sorted(positions.items()):
        if sym in core_symbols:
            if sym not in core_holdings:
                warnings.append(f"{sym} is a core symbol with no core_holdings.json entry -- leaving it untouched")
                continue
            excess = pos.qty - core_holdings[sym]
            if excess > _QTY_EPSILON:
                sells.append((sym, round(excess, 6)))
        elif pos.qty > _QTY_EPSILON:
            sells.append((sym, pos.qty))
    return sells, warnings


def plan_core_topups(
    core_symbols: Sequence[str],
    core_holdings: Dict[str, float],
    prices: Dict[str, float],
    core_pool_usd: float,
    min_notional_usd: float,
) -> List[Tuple[str, float]]:
    """(symbol, usd) buys that bring each core symbol up to an equal share
    of the core pool. Never negative -- an over-target symbol isn't sold."""
    if not core_symbols or core_pool_usd <= 0:
        return []
    target = core_pool_usd / len(core_symbols)
    buys = []
    for sym in core_symbols:
        price = prices.get(sym)
        if not price:
            continue
        shortfall = target - core_holdings.get(sym, 0.0) * price
        if shortfall >= min_notional_usd:
            buys.append((sym, round(shortfall, 2)))
    return buys


def _price(broker: Broker, symbol: str) -> Dict[str, float]:
    q = broker.quote(symbol)
    return {"bid": q.bid, "ask": q.ask, "mid": q.mid}


def _wait_for_fills(broker: Broker, client_ids: List[str], timeout_sec: float) -> Dict[str, object]:
    """Poll until every order is in a final state or the timeout passes.
    Returns client id -> last-seen OrderView (or None if never found)."""
    deadline = time.monotonic() + timeout_sec
    latest: Dict[str, object] = {}
    while True:
        pending = []
        for cid in client_ids:
            order = broker.order_by_client_id(cid)
            latest[cid] = order
            if order is None or order.status in OPEN_ORDER_STATUSES:
                pending.append(cid)
        if not pending or time.monotonic() >= deadline:
            return latest
        print(f"  waiting on {len(pending)} order(s): {', '.join(pending)}")
        time.sleep(10)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true", help="actually trade and write sleeves.json (default: dry run)")
    parser.add_argument("--restart", action="store_true", help="replace an experiment that's already running")
    parser.add_argument("--fill-timeout-sec", type=float, default=600.0)
    args = parser.parse_args()

    cfg = load_config()
    if not cfg.alpaca_paper:
        print("Refusing: this resets positions and is meant for PAPER accounts only (ALPACA_PAPER=false).")
        return 2
    broker = Broker(cfg)
    mode = "EXECUTE" if args.execute else "DRY RUN (nothing will change -- pass --execute to do it)"
    print(f"== start_sleeve_experiment: {mode}")

    existing = load_sleeves(cfg)
    if existing.started_at is not None and not args.restart:
        print(f"An experiment is already running (started {existing.started_at.isoformat()}). "
              "Pass --restart to replace it.")
        return 2

    if args.execute:
        state_path = Path(cfg.state_file_path)
        if state_path.exists():
            try:
                written = datetime.fromisoformat(json.loads(state_path.read_text(encoding="utf-8"))["written_at"])
                age = (datetime.now(timezone.utc) - written).total_seconds()
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                age = None
            if age is not None and age < cfg.loop_interval_sec * 2:
                print(f"Refusing: runtime_state.json was written {age:.0f}s ago -- the engine looks like it's "
                      "still running. Stop it first (e.g. Stop-Service PaperTiger-Engine).")
                return 2
        if not broker.market_open():
            print("Refusing: the market is closed -- the sells and top-ups need it open.")
            return 2

    open_orders = broker.open_orders()
    if open_orders:
        print("Open orders exist -- let them fill or cancel them first:")
        for o in open_orders:
            print(f"  {o.client_order_id} {o.side} {o.symbol} status={o.status}")
        if args.execute:
            return 2

    kinds = cfg.sleeve_kinds()
    sleeve_ids = [SLEEVE_IDS[k] for k in kinds]
    print(f"Sleeves: core + {', '.join(SLEEVE_LABELS[s] for s in sleeve_ids)}  "
          f"(core pool ${cfg.core_pool_usd:,.2f}, each signal sleeve ${cfg.sleeve_pool_usd:,.2f})")

    # -- symbols for each sleeve -------------------------------------------
    try:
        universe = json.loads(Path(cfg.tactical_universe_file_path).read_text(encoding="utf-8"))
        ranked = [s.upper() for s in universe.get("symbols", []) if isinstance(s, str)]
    except (OSError, json.JSONDecodeError, AttributeError):
        ranked = []
    if not ranked:
        print(f"No ranked symbols in {cfg.tactical_universe_file_path} -- run "
              "scripts/refresh_tactical_universe.py first.")
        return 2
    print(f"Ranked universe: {len(ranked)} symbols (generated {universe.get('generated_at', '?')})")
    sectors = load_candidate_sectors(cfg.candidate_universe_file_path)
    dealt = deal_symbols(ranked, sectors, sleeve_ids, exclude=cfg.symbols)
    for sid, syms in dealt.items():
        mix: Dict[str, int] = {}
        for s in syms:
            mix[sectors.get(s, "other")] = mix.get(sectors.get(s, "other"), 0) + 1
        mix_text = ", ".join(f"{k} {v}" for k, v in sorted(mix.items(), key=lambda kv: -kv[1]))
        print(f"  {SLEEVE_LABELS[sid]:<16} {len(syms):>2} symbols: {' '.join(syms)}")
        print(f"  {'':<16}    sectors: {mix_text}")

    # -- step 1: sell everything that isn't core -----------------------------
    allocator = CoreAllocator(cfg)
    core_holdings = allocator.load()
    positions = broker.positions()
    sells, warnings = plan_reset_sells(positions, cfg.symbols, core_holdings)
    for w in warnings:
        print(f"  WARNING: {w}")
    today = datetime.now(timezone.utc).date()
    print(f"Step 1 -- sell {len(sells)} non-core position(s):")
    for sym, qty in sells:
        print(f"  sell {qty} {sym} (~${qty * positions[sym].current_price:,.2f})")

    # -- step 2: top up core ------------------------------------------------
    core_prices = {}
    for sym in cfg.symbols:
        try:
            core_prices[sym] = _price(broker, sym)["mid"]
        except BrokerError as e:
            print(f"  WARNING: no quote for core symbol {sym}: {e}")
    topups = plan_core_topups(cfg.symbols, core_holdings, core_prices, cfg.core_pool_usd, cfg.min_notional_usd)
    print(f"Step 2 -- top core up to ${cfg.core_pool_usd:,.2f} "
          f"(${cfg.core_pool_usd / max(1, len(cfg.symbols)):,.2f} per symbol):")
    for sym, usd in topups:
        print(f"  buy ${usd:,.2f} of {sym}")
    if not topups:
        print("  nothing to top up")

    needed = sum(usd for _, usd in topups) + cfg.sleeve_pool_usd * len(sleeve_ids)
    account = broker.account()
    print(f"Money needed (top-ups + signal pools): ${needed:,.2f}; account buying power ${account.buying_power:,.2f}"
          " (sales from step 1 add to that)")

    if not args.execute:
        print("Dry run complete -- nothing was changed.")
        return 0

    # -- execute step 1 -------------------------------------------------------
    sell_ids = []
    for sym, qty in sells:
        cid = reset_client_order_id(sym, today)
        sell_ids.append(cid)
        if broker.order_by_client_id(cid) is not None:
            continue  # placed by an earlier run today -- just wait on it
        bid = _price(broker, sym)["bid"] or positions[sym].current_price
        broker.submit_limit_sell(sym, qty, bid * (1 - _NUDGE), cid)
        print(f"  submitted sell {qty} {sym}")
    results = _wait_for_fills(broker, sell_ids, args.fill_timeout_sec)
    unfilled = [cid for cid, o in results.items() if o is None or o.status != "filled"]
    if unfilled:
        print(f"Not every sell filled ({', '.join(unfilled)}). Nothing else was done; re-run once they "
              "settle (or cancel them).")
        return 1

    # -- execute step 2 -------------------------------------------------------
    topup_ids = []
    for sym, usd in topups:
        cid = core_topup_client_order_id(sym, today)
        if broker.order_by_client_id(cid) is not None:
            print(f"  {sym} top-up already placed today -- not repeating it")
            continue
        topup_ids.append((sym, cid))
        broker.submit_limit_buy(sym, usd, _price(broker, sym)["ask"] * (1 + _NUDGE), cid)
        print(f"  submitted core top-up ${usd:,.2f} of {sym}")
    results = _wait_for_fills(broker, [cid for _, cid in topup_ids], args.fill_timeout_sec)
    for sym, cid in topup_ids:
        order = results.get(cid)
        if order is not None and order.filled_qty:
            allocator.add_to_holding(sym, order.filled_qty)
            print(f"  recorded {order.filled_qty} more core {sym}")
    unfilled = [cid for _, cid in topup_ids if results.get(cid) is None or results[cid].status != "filled"]
    if unfilled:
        print(f"Not every top-up filled ({', '.join(unfilled)}). Re-run once they settle; filled parts are "
              "already recorded.")
        return 1

    # -- step 3: write sleeves.json ------------------------------------------
    core_holdings = allocator.load()
    positions = broker.positions()
    core_value = sum(
        core_holdings.get(sym, 0.0) * positions[sym].current_price for sym in cfg.symbols if sym in positions
    )
    start_prices: Dict[str, float] = {}
    for syms in dealt.values():
        for sym in syms:
            try:
                start_prices[sym] = round(_price(broker, sym)["mid"], 4)
            except BrokerError as e:
                print(f"  WARNING: no start price for {sym} ({e}); it won't count toward its benchmark")
    started_at = datetime.now(timezone.utc)
    history = Path(cfg.sleeve_history_file_path)
    if history.exists():
        archived = history.with_name(f"{history.stem}.until-{started_at.date().isoformat()}{history.suffix}")
        history.replace(archived)
        print(f"  archived the previous sleeve history to {archived.name}")
    write_sleeves_file(
        cfg.sleeves_file_path,
        started_at=started_at,
        core_pool_usd=cfg.core_pool_usd,
        core_start_cash=cfg.core_pool_usd - core_value,
        core_symbols=cfg.symbols,
        sleeves={
            sid: {
                "signal_kind": kind,
                "pool_usd": cfg.sleeve_pool_usd,
                "symbols": dealt[sid],
                "start_prices": {s: start_prices[s] for s in dealt[sid] if s in start_prices},
            }
            for sid, kind in zip(sleeve_ids, kinds)
        },
    )
    print(f"Wrote {cfg.sleeves_file_path}: experiment started {started_at.isoformat()}. "
          "Start the engine (e.g. Start-Service PaperTiger-Engine).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
