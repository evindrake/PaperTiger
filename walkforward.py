"""
walkforward.py -- out-of-sample validation via sliding train/test windows.

Why this file exists at all: a single backtest.py run tells you how one
strategy configuration did on one historical path. It says nothing about
whether that configuration would have looked good on ANY historical data
purely by chance (overfitting), or whether the "best" parameters are stable
enough to trust going forward. Walk-forward analysis is the standard way to
at least partially answer that:

  1. Slide a TRAIN window across history. On each TRAIN window, try every
     combination in a small parameter grid (fast/slow SMA pairs) and pick
     the single best one by Sharpe ratio -- using ONLY that window's data.
  2. Apply ONLY that chosen configuration to the following TEST window
     (which the parameter search never saw) and record how it did.
  3. Slide both windows forward and repeat, compounding capital across the
     TEST folds so the stitched-together OOS (out-of-sample) equity curve
     represents "what would have actually happened" if you re-tuned on a
     rolling basis and only ever traded the next chunk of unseen data.

The verdict this script cares most about is the GAP between in-sample
(TRAIN) performance and out-of-sample (TEST) performance. A strategy that
looks great in-sample but mediocre or worse out-of-sample is the textbook
signature of overfitting -- the whole point of this file is to make that
gap impossible to miss.

This reuses backtest.py's Portfolio/run_backtest machinery directly (same
caps, same fill/no-lookahead rules, same signals.py) rather than
reimplementing a second simulator -- one honest simulation path, not two
that could quietly disagree.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Tuple

from backtest import Bar, BacktestConfig, compute_metrics, load_csv_bars, run_backtest, run_buy_and_hold_benchmark

# Small, deliberately modest parameter grids, one per signal kind. This is
# NOT an attempt to find "the best" configuration -- a bigger grid searched
# more aggressively is just a fancier way to overfit. Small and boring, on
# purpose. Which grid is used is picked automatically from cfg.signal_kind
# (see _default_param_grid), so switching SIGNAL_KIND in .env is enough to
# walk-forward-test the other signal without touching this file.
_SMA_PARAM_GRID: List[dict] = [
    {"signal_fast": 5, "signal_slow": 20},
    {"signal_fast": 10, "signal_slow": 30},
    {"signal_fast": 10, "signal_slow": 50},
    {"signal_fast": 20, "signal_slow": 60},
    {"signal_fast": 5, "signal_slow": 40},
    {"signal_fast": 15, "signal_slow": 45},
]

_RSI_PARAM_GRID: List[dict] = [
    {"signal_period": 7, "signal_oversold": 30.0, "signal_overbought": 70.0},
    {"signal_period": 14, "signal_oversold": 30.0, "signal_overbought": 70.0},
    {"signal_period": 14, "signal_oversold": 25.0, "signal_overbought": 75.0},
    {"signal_period": 21, "signal_oversold": 30.0, "signal_overbought": 70.0},
]

# NOTE on ml_classifier: unlike the two grids above, this one does NOT
# retrain the underlying model per fold -- train_ml_signal.py trains ONE
# model file, once, ahead of time, and this grid only sweeps the buy/sell
# probability THRESHOLDS applied on top of that fixed model's output. A
# true walk-forward test of the model itself (retraining fresh on each
# TRAIN fold) is future work -- see backtest.py's LIMITATIONS block.
_ML_PARAM_GRID: List[dict] = [
    {"signal_ml_buy_threshold": 0.55, "signal_ml_sell_threshold": 0.45},
    {"signal_ml_buy_threshold": 0.60, "signal_ml_sell_threshold": 0.40},
    {"signal_ml_buy_threshold": 0.65, "signal_ml_sell_threshold": 0.35},
    {"signal_ml_buy_threshold": 0.52, "signal_ml_sell_threshold": 0.48},
]


def _default_param_grid(cfg) -> List[dict]:
    if cfg.signal_kind == "rsi_reversion":
        return _RSI_PARAM_GRID
    if cfg.signal_kind == "ml_classifier":
        return _ML_PARAM_GRID
    return _SMA_PARAM_GRID


@dataclass(frozen=True)
class Fold:
    train_start: date
    train_end: date
    test_start: date
    test_end: date


def _make_folds(all_dates: List[date], train_days: int, test_days: int, step_days: int) -> List[Fold]:
    folds = []
    i = 0
    while i + train_days + test_days <= len(all_dates):
        train_start = all_dates[i]
        train_end = all_dates[i + train_days - 1]
        test_start = all_dates[i + train_days]
        test_end = all_dates[i + train_days + test_days - 1]
        folds.append(Fold(train_start, train_end, test_start, test_end))
        i += step_days
    return folds


def _slice_bars(bars_by_symbol: Dict[str, List[Bar]], start: date, end: date) -> Dict[str, List[Bar]]:
    return {
        symbol: [b for b in bars if start <= b.trade_date <= end]
        for symbol, bars in bars_by_symbol.items()
    }


def _cfg_with_params(cfg, params: dict):
    """config.Config is frozen -- return a shallow copy with just the given
    signal params swapped, via dataclasses.replace (works fine here since
    Config is itself a dataclass). `params` keys are whichever signal fields
    the current grid varies (signal_fast/signal_slow for sma_crossover,
    signal_period/signal_oversold/signal_overbought for rsi_reversion)."""
    return replace(cfg, **params)


def run_walkforward(
    cfg,
    bt_cfg: BacktestConfig,
    bars_by_symbol: Dict[str, List[Bar]],
    train_days: int = 180,
    test_days: int = 60,
    step_days: int = 60,
    param_grid: List[dict] = None,
) -> dict:
    param_grid = param_grid or _default_param_grid(cfg)

    common_dates = None
    for bars in bars_by_symbol.values():
        dates = {b.trade_date for b in bars}
        common_dates = dates if common_dates is None else (common_dates & dates)
    all_dates = sorted(common_dates)

    folds = _make_folds(all_dates, train_days, test_days, step_days)
    if not folds:
        raise ValueError(
            f"not enough data ({len(all_dates)} common trading days) for even one "
            f"train({train_days})+test({test_days}) fold"
        )

    fold_reports = []
    stitched_oos_curve: List[Tuple[date, float]] = []
    # A buy-and-hold benchmark stitched over the SAME test folds, compounded
    # the same way -- this is what the dashboard overlays on the OOS curve
    # so "the line went up" and "the line beat buy-and-hold" are visually
    # distinguishable instead of easy to conflate.
    stitched_benchmark_curve: List[Tuple[date, float]] = []
    running_capital = bt_cfg.initial_capital
    running_benchmark_capital = bt_cfg.initial_capital
    chosen_params_seen: List[dict] = []

    for fold in folds:
        train_bars = _slice_bars(bars_by_symbol, fold.train_start, fold.train_end)
        test_bars = _slice_bars(bars_by_symbol, fold.test_start, fold.test_end)

        # 1. Sweep the grid on TRAIN only, pick best by Sharpe.
        best_params = None
        best_sharpe = float("-inf")
        best_train_metrics = None
        for params in param_grid:
            trial_cfg = _cfg_with_params(cfg, params)
            try:
                result = run_backtest(trial_cfg, bt_cfg, train_bars)
            except ValueError:
                continue  # not enough warmup room in this window for this param combo
            sharpe = result["metrics"]["sharpe"]
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = params
                best_train_metrics = result["metrics"]

        if best_params is None:
            # No param combination had enough warmup room in this TRAIN
            # window -- skip the fold rather than guessing a configuration.
            continue

        chosen_params_seen.append(best_params)

        # 2. Apply ONLY the chosen params to TEST (never re-tuned on test data).
        test_cfg = _cfg_with_params(cfg, best_params)
        test_bt_cfg = replace(bt_cfg, initial_capital=running_capital)
        try:
            test_result = run_backtest(test_cfg, test_bt_cfg, test_bars)
        except ValueError as e:
            fold_reports.append({
                "train_start": fold.train_start.isoformat(), "train_end": fold.train_end.isoformat(),
                "test_start": fold.test_start.isoformat(), "test_end": fold.test_end.isoformat(),
                "chosen_params": best_params,
                "train_metrics": best_train_metrics, "test_metrics": None,
                "skipped_reason": str(e),
            })
            continue

        test_metrics = test_result["metrics"]
        fold_reports.append({
            "train_start": fold.train_start.isoformat(), "train_end": fold.train_end.isoformat(),
            "test_start": fold.test_start.isoformat(), "test_end": fold.test_end.isoformat(),
            "chosen_params": best_params,
            "train_metrics": best_train_metrics,
            "test_metrics": test_metrics,
        })

        # Compound capital forward into the next fold's TEST starting point.
        running_capital = test_metrics["final_equity"]
        stitched_oos_curve.extend(
            (date.fromisoformat(pt["date"]), pt["equity"]) for pt in test_result["equity_curve"]
        )

        # Same stitching, same compounding, for the buy-and-hold benchmark
        # over this fold's TEST window -- kept as a separate running capital
        # so the two curves are independently comparable start to finish.
        try:
            benchmark_bt_cfg = replace(bt_cfg, initial_capital=running_benchmark_capital)
            benchmark_result = run_buy_and_hold_benchmark(benchmark_bt_cfg, test_bars)
            running_benchmark_capital = benchmark_result["metrics"]["final_equity"]
            stitched_benchmark_curve.extend(
                (date.fromisoformat(pt["date"]), pt["equity"]) for pt in benchmark_result["equity_curve"]
            )
        except (ValueError, ZeroDivisionError):
            pass  # benchmark couldn't be computed for this fold -- OOS curve itself is unaffected

    completed = [f for f in fold_reports if f.get("test_metrics") is not None]
    if not completed:
        raise ValueError("no fold produced a usable test result -- try a shorter train/test window or more data")

    avg_train_return = statistics.mean(f["train_metrics"]["total_return"] for f in completed)
    avg_test_return = statistics.mean(f["test_metrics"]["total_return"] for f in completed)
    overfit_gap = avg_train_return - avg_test_return

    stitched_metrics = compute_metrics(stitched_oos_curve, num_trades=sum(f["test_metrics"]["num_trades"] for f in completed))
    stitched_benchmark_metrics = (
        compute_metrics(stitched_benchmark_curve, num_trades=0) if len(stitched_benchmark_curve) >= 2 else None
    )
    beats_buy_and_hold = (
        stitched_metrics["total_return"] > stitched_benchmark_metrics["total_return"]
        if stitched_benchmark_metrics is not None else None
    )

    # Dicts aren't hashable -- compare by their sorted (key, value) tuples instead.
    unique_params = {tuple(sorted(p.items())) for p in chosen_params_seen}
    params_stable = len(unique_params) == 1

    # Beating buy-and-hold is the FIRST bar this project cares about (see
    # README) -- state it plainly before anything about overfitting, so
    # "the OOS line went up" and "the OOS line beat the passive benchmark"
    # never get conflated into "sounds good."
    if beats_buy_and_hold is False:
        bh_clause = (
            f"the stitched out-of-sample equity LOST to buy-and-hold "
            f"({stitched_metrics['total_return']:+.2%} vs {stitched_benchmark_metrics['total_return']:+.2%}) -- "
        )
    elif beats_buy_and_hold is True:
        bh_clause = (
            f"the stitched out-of-sample equity beat buy-and-hold "
            f"({stitched_metrics['total_return']:+.2%} vs {stitched_benchmark_metrics['total_return']:+.2%}), but "
        )
    else:
        bh_clause = ""

    if overfit_gap > 0.10:
        verdict = (
            f"{bh_clause}there's a large in-sample vs out-of-sample gap (>10 percentage points): "
            "the parameter sweep is likely overfitting to each training window and should not be trusted."
        )
    elif avg_test_return <= 0:
        verdict = f"{bh_clause}out-of-sample return is flat or negative on average: no evidence of a forward edge."
    elif not params_stable:
        verdict = (
            f"{bh_clause}out-of-sample returns are positive, but the 'best' parameters kept changing "
            "between folds -- that instability is itself a warning sign, not a strategy to trust."
        )
    elif beats_buy_and_hold is False:
        verdict = (
            f"{bh_clause}out-of-sample returns were otherwise positive and stable, but losing to "
            "buy-and-hold is the bar that matters most -- this configuration doesn't clear it."
        )
    else:
        verdict = (
            f"{bh_clause}out-of-sample returns are positive and the chosen parameters were stable "
            "across folds. That's a mildly encouraging sign, not proof of a real edge -- this is "
            "still one historical path on a placeholder textbook signal."
        )

    return {
        "folds": fold_reports,
        "avg_in_sample_return": avg_train_return,
        "avg_out_of_sample_return": avg_test_return,
        "overfit_gap": overfit_gap,
        "stitched_oos_metrics": stitched_metrics,
        "stitched_oos_equity_curve": [{"date": d.isoformat(), "equity": v} for d, v in stitched_oos_curve],
        "stitched_benchmark_metrics": stitched_benchmark_metrics,
        "stitched_benchmark_equity_curve": [{"date": d.isoformat(), "equity": v} for d, v in stitched_benchmark_curve],
        "beats_buy_and_hold": beats_buy_and_hold,
        "chosen_params_per_fold": chosen_params_seen,
        "params_stable": params_stable,
        "verdict": verdict,
    }


def main() -> None:
    from config import load_config

    parser = argparse.ArgumentParser(description="Run walk-forward validation for PaperTiger.")
    parser.add_argument("--source", choices=["csv", "alpaca"], default="csv")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--start", default=None, help="YYYY-MM-DD, only used with --source alpaca")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD, only used with --source alpaca")
    parser.add_argument("--train-days", type=int, default=180)
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument("--step-days", type=int, default=60)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--commission-usd", type=float, default=0.0)
    parser.add_argument("--out", default="walkforward_results.json")
    args = parser.parse_args()

    cfg = load_config()
    bt_cfg = BacktestConfig(
        initial_capital=cfg.seed_usd,
        slippage_bps=args.slippage_bps,
        commission_usd=args.commission_usd,
    )

    if args.source == "csv":
        bars_by_symbol = {symbol: load_csv_bars(symbol, args.data_dir) for symbol in cfg.symbols}
    else:
        from datetime import timezone as _timezone

        from backtest import load_alpaca_bars

        if not args.start or not args.end:
            raise SystemExit("--start and --end are required with --source alpaca")
        start = datetime.fromisoformat(args.start).replace(tzinfo=_timezone.utc)
        end = datetime.fromisoformat(args.end).replace(tzinfo=_timezone.utc)
        bars_by_symbol = {symbol: load_alpaca_bars(symbol, start, end, cfg) for symbol in cfg.symbols}

    result = run_walkforward(
        cfg, bt_cfg, bars_by_symbol,
        train_days=args.train_days, test_days=args.test_days, step_days=args.step_days,
    )

    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Folds completed: {len(result['folds'])}")
    print(f"Avg in-sample (TRAIN) return:  {result['avg_in_sample_return']:+.2%}")
    print(f"Avg out-of-sample (TEST) return: {result['avg_out_of_sample_return']:+.2%}")
    print(f"Overfit gap (train - test):    {result['overfit_gap']:+.2%}")
    print(f"Chosen params stable across folds: {result['params_stable']}")
    print(f"Stitched OOS Sharpe: {result['stitched_oos_metrics']['sharpe']:.2f}  "
          f"MaxDD: {result['stitched_oos_metrics']['max_drawdown']:.2%}")
    if result["stitched_benchmark_metrics"] is not None:
        print(f"Stitched OOS total return: {result['stitched_oos_metrics']['total_return']:+.2%}  "
              f"vs Buy&Hold: {result['stitched_benchmark_metrics']['total_return']:+.2%}  "
              f"(beats buy-and-hold: {result['beats_buy_and_hold']})")
    print()
    print(f"VERDICT: {result['verdict']}")
    print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
