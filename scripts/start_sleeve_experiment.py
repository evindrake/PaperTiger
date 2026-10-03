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
     them start equal. The settings in effect (risk profile, signal
     parameters, breaker action) are recorded too, so the dashboard can
     flag any change made partway through.
  4. Research, so there's nothing else to remember: train the ML sleeve's
     model on its own stocks (if ML is one of the sleeves), then run a
     walk-forward test of every sleeve on its own stocks -- a historical
     baseline for the live comparison, shown on the Compare tab. The
     nightly research task keeps both up to date after that. A failure
     here doesn't undo steps 1-3; it's reported, and --research-only
     reruns just this step. (--skip-research leaves it out.)

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
    python scripts/start_sleeve_experiment.py --research-only       # just rerun step 4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from broker import Broker, BrokerError  # noqa: E402
from config import load_config  # noqa: E402
from core import CoreAllocator  # noqa: E402
from refresh_tactical_universe import load_candidate_sectors  # noqa: E402
from safety import RiskProfileStore  # noqa: E402
from sleeves import (  # noqa: E402
    OPEN_ORDER_STATUSES,
    SLEEVE_IDS,
    SLEEVE_LABELS,
    core_topup_client_order_id,
    deal_symbols,
    experiment_settings,
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


def _pid_alive(pid: int) -> bool:
    """Whether a process with this id is running. Never signals it."""
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    import os

    try:
        os.kill(pid, 0)  # signal 0 = existence check only on POSIX
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def engine_looks_running(state_path: Path, loop_interval_sec: float) -> bool:
    """True if the engine that last wrote runtime_state.json is still
    running. Uses the process id it records; an older state file without
    one falls back to "written within the last two ticks"."""
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    pid = state.get("engine_pid")
    if isinstance(pid, int):
        return _pid_alive(pid)
    try:
        written = datetime.fromisoformat(state["written_at"])
    except (KeyError, TypeError, ValueError):
        return False
    return (datetime.now(timezone.utc) - written).total_seconds() < loop_interval_sec * 2


def run_research(cfg, sleeve_ids: Sequence[str]) -> bool:
    """Step 4: train the ML sleeve's model on its own stocks, then
    walk-forward-test every sleeve on its own stocks. Same 4-year window as
    the nightly research task. Returns False if anything failed."""
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=4 * 365)
    window = ["--source", "alpaca", "--start", start.isoformat(), "--end", end.isoformat()]
    steps = []
    if "ml" in sleeve_ids:
        steps.append(("train the ML sleeve's model on its own stocks",
                      ["train_ml_signal.py", *window, "--sleeve", "ml", "--model-out", cfg.signal_model_path]))
    steps.append(("walk-forward test of every sleeve on its own stocks",
                  ["walkforward.py", *window, "--all-sleeves"]))
    ok = True
    for description, cmd in steps:
        print(f"Step 4 -- {description} (this can take a few minutes)...")
        result = subprocess.run([sys.executable, *cmd], cwd=PROJECT_DIR)
        if result.returncode != 0:
            print(f"  that step failed (exit {result.returncode}) -- rerun with --research-only")
            ok = False
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true", help="actually trade and write sleeves.json (default: dry run)")
    parser.add_argument("--restart", action="store_true", help="replace an experiment that's already running")
    parser.add_argument("--skip-research", action="store_true", help="don't train/test after starting (step 4)")
    parser.add_argument("--research-only", action="store_true",
                        help="only rerun step 4 for the experiment that's already running")
    parser.add_argument("--fill-timeout-sec", type=float, default=600.0)
    args = parser.parse_args()

    cfg = load_config()
    if not cfg.alpaca_paper:
        print("Refusing: this resets positions and is meant for PAPER accounts only (ALPACA_PAPER=false).")
        return 2

    if args.research_only:
        running = load_sleeves(cfg)
        if running.started_at is None:
            print("No experiment is running yet -- nothing to research.")
            return 2
        return 0 if run_research(cfg, [s.sleeve_id for s in running.sleeves if s.symbols]) else 1

    broker = Broker(cfg)
    mode = "EXECUTE" if args.execute else "DRY RUN (nothing will change -- pass --execute to do it)"
    print(f"== start_sleeve_experiment: {mode}")

    existing = load_sleeves(cfg)
    if existing.started_at is not None and not args.restart:
        print(f"An experiment is already running (started {existing.started_at.isoformat()}). "
              "Pass --restart to replace it.")
        return 2

    if args.execute:
        if engine_looks_running(Path(cfg.state_file_path), cfg.loop_interval_sec):
            print("Refusing: the engine is still running. Stop it first (Windows: Stop-Service "
                  "PaperTiger-Engine; Linux: systemctl --user stop papertiger-engine; macOS: "
                  "launchctl unload ~/Library/LaunchAgents/com.papertiger.engine.plist).")
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
    profile_state = RiskProfileStore(cfg.risk_profile_file_path).load()
    settings = experiment_settings(replace(cfg, **profile_state.resolve()), profile_state.profile)
    print(f"Settings recorded at the start: risk profile {settings['risk_profile']}, trade size "
          f"{settings['trade_size_pct']:.0%} of each pool, up to {settings['max_open_positions']} positions, "
          f"circuit breaker action {settings['breaker_action']}")

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
        settings=settings,
    )
    print(f"Wrote {cfg.sleeves_file_path}: experiment started {started_at.isoformat()}, "
          f"risk profile {settings['risk_profile']}.")

    # -- step 4: research ------------------------------------------------------
    research_ok = True
    if not args.skip_research:
        research_ok = run_research(cfg, sleeve_ids)
    print("Done. Start the engine (Windows: Start-Service PaperTiger-Engine; Linux: systemctl --user start "
          "papertiger-engine; macOS: launchctl load ~/Library/LaunchAgents/com.papertiger.engine.plist).")
    return 0 if research_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
