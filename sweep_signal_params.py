"""
sweep_signal_params.py -- quick exploration tool, NOT a validation tool.

Runs backtest.py's simulator once per parameter combination across the
FULL requested date range and prints a sorted comparison table. This is
useful for getting a fast read on "does anything in this neighborhood look
interesting at all" -- but unlike walkforward.py, it has no train/test
split, so picking "the best" row here and believing it will repeat
out-of-sample is exactly the overfitting mistake this project's philosophy
warns against. Treat this as a first filter, not a conclusion -- anything
that looks good here should still go through walkforward.py before you
trust it at all, and even a strong walk-forward pass on this small a
universe/date range is still not proof of a real edge.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from backtest import Bar, BacktestConfig, load_alpaca_bars, load_csv_bars, run_backtest, run_buy_and_hold_benchmark

_SMA_SWEEP: List[dict] = [
    {"signal_kind": "sma_crossover", "signal_fast": 3, "signal_slow": 15},
    {"signal_kind": "sma_crossover", "signal_fast": 5, "signal_slow": 20},
    {"signal_kind": "sma_crossover", "signal_fast": 10, "signal_slow": 30},
    {"signal_kind": "sma_crossover", "signal_fast": 10, "signal_slow": 50},
    {"signal_kind": "sma_crossover", "signal_fast": 20, "signal_slow": 60},
    {"signal_kind": "sma_crossover", "signal_fast": 5, "signal_slow": 40},
    {"signal_kind": "sma_crossover", "signal_fast": 15, "signal_slow": 45},
]

_RSI_SWEEP: List[dict] = [
    {"signal_kind": "rsi_reversion", "signal_period": 7, "signal_oversold": 30.0, "signal_overbought": 70.0},
    {"signal_kind": "rsi_reversion", "signal_period": 10, "signal_oversold": 20.0, "signal_overbought": 80.0},
    {"signal_kind": "rsi_reversion", "signal_period": 14, "signal_oversold": 30.0, "signal_overbought": 70.0},
    {"signal_kind": "rsi_reversion", "signal_period": 14, "signal_oversold": 25.0, "signal_overbought": 75.0},
    {"signal_kind": "rsi_reversion", "signal_period": 21, "signal_oversold": 30.0, "signal_overbought": 70.0},
]


def _label(params: dict) -> str:
    if params["signal_kind"] == "sma_crossover":
        return f"sma({params['signal_fast']}/{params['signal_slow']})"
    return f"rsi({params['signal_period']}, {params['signal_oversold']:.0f}/{params['signal_overbought']:.0f})"


def main() -> None:
    from config import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["csv", "alpaca"], default="csv")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--grid", choices=["sma", "rsi", "both"], default="both")
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--commission-usd", type=float, default=0.0)
    parser.add_argument("--out", default=None, help="optional path to write full JSON results")
    args = parser.parse_args()

    cfg = load_config()
    bt_cfg = BacktestConfig(
        initial_capital=cfg.seed_usd,
        slippage_bps=args.slippage_bps,
        commission_usd=args.commission_usd,
    )

    bars_by_symbol: Dict[str, List[Bar]] = {}
    if args.source == "csv":
        for symbol in cfg.symbols:
            bars_by_symbol[symbol] = load_csv_bars(symbol, args.data_dir)
    else:
        if not args.start or not args.end:
            raise SystemExit("--start and --end are required with --source alpaca")
        start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
        end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
        for symbol in cfg.symbols:
            bars_by_symbol[symbol] = load_alpaca_bars(symbol, start, end, cfg)

    grid = {"sma": _SMA_SWEEP, "rsi": _RSI_SWEEP, "both": _SMA_SWEEP + _RSI_SWEEP}[args.grid]

    benchmark = run_buy_and_hold_benchmark(bt_cfg, bars_by_symbol)
    bench_return = benchmark["metrics"]["total_return"]

    rows = []
    for params in grid:
        trial_cfg = replace(cfg, **params)
        try:
            result = run_backtest(trial_cfg, bt_cfg, bars_by_symbol)
        except ValueError as e:
            rows.append({"label": _label(params), "params": params, "error": str(e)})
            continue
        m = result["metrics"]
        rows.append({
            "label": _label(params),
            "params": params,
            "total_return": m["total_return"],
            "cagr": m["cagr"],
            "max_drawdown": m["max_drawdown"],
            "sharpe": m["sharpe"],
            "num_trades": m["num_trades"],
            "beats_buy_and_hold": m["total_return"] > bench_return,
        })

    ok_rows = [r for r in rows if "error" not in r]
    ok_rows.sort(key=lambda r: r["sharpe"], reverse=True)

    print(f"Buy & Hold benchmark: total return {bench_return:+.2%}, CAGR {benchmark['metrics']['cagr']:+.2%}\n")
    print(f"{'Config':<24} {'Return':>9} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>7} {'Trades':>7}  Beats B&H")
    for r in ok_rows:
        print(
            f"{r['label']:<24} {r['total_return']:>+8.2%} {r['cagr']:>+7.2%} "
            f"{r['max_drawdown']:>7.2%} {r['sharpe']:>7.2f} {r['num_trades']:>7}  "
            f"{'YES' if r['beats_buy_and_hold'] else 'no'}"
        )
    for r in rows:
        if "error" in r:
            print(f"{r['label']:<24} SKIPPED: {r['error']}")

    winners = [r for r in ok_rows if r["beats_buy_and_hold"]]
    print()
    if winners:
        print(f"{len(winners)} of {len(ok_rows)} configs beat buy-and-hold on this single run.")
        print("Reminder: this is an in-sample sweep with no train/test split -- run walkforward.py on any")
        print("candidate before trusting it. Picking 'the best' row here is how overfitting happens.")
    else:
        print("No configuration beat buy-and-hold on this run.")

    if args.out:
        Path(args.out).write_text(json.dumps({"benchmark": benchmark["metrics"], "rows": rows}, indent=2), encoding="utf-8")
        print(f"\nFull results written to {args.out}")


if __name__ == "__main__":
    main()
