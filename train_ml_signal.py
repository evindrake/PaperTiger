"""
train_ml_signal.py -- trains the model ml_signal.py's "ml_classifier" signal
loads at decision time.

============================== LIMITATIONS ==============================
  - SMALL, NOISY DATASET. A few years of daily bars across 4 correlated
    ETFs is a tiny, low-signal dataset by machine-learning standards. High
    train accuracy here is easy and means very little; only the
    CHRONOLOGICAL holdout accuracy and (more importantly) the downstream
    backtest/walk-forward economic result are worth trusting at all.
  - LABELS ARE NOISY BY CONSTRUCTION. "Will price be higher in N days" on a
    single, mostly-correlated basket of US equity ETFs is dominated by one
    shared macro trend, not N independent bets -- don't expect this to
    generalize to a different market regime than it was trained on.
  - The train/test split here is CHRONOLOGICAL (earliest data = train,
    latest = test), never shuffled -- shuffling would leak future
    information into training and make the reported accuracy meaningless.
  - This script reports classification accuracy, which is NOT the same as
    trading performance. A model can have >50% directional accuracy and
    still lose money after transaction costs and cap enforcement, or have
    <50% accuracy and still make money if its correct calls are sized
    right. ALWAYS check the accompanying backtest.py/walkforward.py run
    (this script does one automatically, see below) before believing a
    model "works."
===========================================================================

Usage:
    python train_ml_signal.py --source csv --horizon 5
    python train_ml_signal.py --source alpaca --start 2022-01-01 --end 2026-08-22

Writes the trained model to --model-out (default ml_model.joblib) as an
ml_signal.MLModelBundle, and also runs backtest.py's simulator over the
chronological test split so you see an honest read on whether the
classifier's accuracy actually translates into a trading edge, in the same
terms as the other two signals.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from typing import Dict, List, Tuple

from backtest import Bar, BacktestConfig, load_alpaca_bars, load_csv_bars, run_backtest, run_buy_and_hold_benchmark
from ml_signal import FEATURE_NAMES, MLModelBundle, compute_features


def build_dataset(bars: List[Bar], horizon_days: int) -> Tuple[List[List[float]], List[int]]:
    """One (features, label) row per trading day with enough history AND
    enough future days left to know the label. label = 1 if close[i +
    horizon_days] > close[i], else 0."""
    closes = [b.close for b in bars]
    X: List[List[float]] = []
    y: List[int] = []
    for i in range(len(closes)):
        if i + horizon_days >= len(closes):
            break  # no future close far enough ahead to label this row
        features = compute_features(closes[: i + 1])
        if features is None:
            continue  # not enough history yet for this row
        label = 1 if closes[i + horizon_days] > closes[i] else 0
        X.append(features)
        y.append(label)
    return X, y


def main() -> None:
    from config import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["csv", "alpaca"], default="csv")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--horizon", type=int, default=5, help="trading days ahead to predict")
    parser.add_argument("--test-frac", type=float, default=0.25, help="fraction of EACH symbol's data held out, chronologically, for testing")
    parser.add_argument("--model-type", choices=["logistic", "gradient_boosting"], default="logistic")
    parser.add_argument("--model-out", default="ml_model.joblib")
    args = parser.parse_args()

    cfg = load_config()

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

    # Pool training rows across all symbols (more examples from a short date
    # range), but split each symbol's OWN timeline chronologically first --
    # pooling after the split, not before, keeps test rows strictly later
    # in time than the train rows they came from, symbol by symbol.
    X_train, y_train, X_test, y_test = [], [], [], []
    test_bars_by_symbol: Dict[str, List[Bar]] = {}
    for symbol, bars in bars_by_symbol.items():
        X, y = build_dataset(bars, args.horizon)
        if not X:
            print(f"[train_ml_signal] WARNING: no usable rows for {symbol}, skipping")
            continue
        split_idx = int(len(X) * (1 - args.test_frac))
        X_train.extend(X[:split_idx])
        y_train.extend(y[:split_idx])
        X_test.extend(X[split_idx:])
        y_test.extend(y[split_idx:])
        # Also carve out the corresponding TAIL of raw bars (by date, not by
        # feature-row index) so we can run an honest backtest over the same
        # chronological test period below.
        test_bars_by_symbol[symbol] = bars[max(0, len(bars) - int(len(bars) * args.test_frac)):]

    if not X_train or not X_test:
        raise SystemExit("not enough data to build both a train and a test split -- use more history or a smaller --test-frac")

    if args.model_type == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    else:
        from sklearn.ensemble import GradientBoostingClassifier

        model = GradientBoostingClassifier(n_estimators=100, max_depth=2, random_state=0)

    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    baseline = max(sum(y_test), len(y_test) - sum(y_test)) / len(y_test)  # "always predict majority class"

    print(f"Train accuracy: {train_acc:.3f} ({len(X_train)} rows)")
    print(f"Test accuracy:  {test_acc:.3f} ({len(X_test)} rows)  [majority-class baseline: {baseline:.3f}]")
    if test_acc <= baseline + 0.02:
        print(
            "WARNING: test accuracy barely beats (or loses to) the naive 'always guess the "
            "majority class' baseline. This model likely has no real predictive signal."
        )

    bundle = MLModelBundle(model=model, feature_names=FEATURE_NAMES, horizon_days=args.horizon)
    import joblib

    joblib.dump(bundle, args.model_out)
    print(f"Model written to {args.model_out}")

    # Honest downstream check: does this classifier's accuracy translate
    # into a trading edge over the same chronological test window?
    print("\nRunning backtest.py over the chronological test window (same rules/caps as live)...")
    ml_cfg = replace(
        cfg, signal_kind="ml_classifier", signal_model_path=args.model_out,
        signal_ml_buy_threshold=cfg.signal_ml_buy_threshold, signal_ml_sell_threshold=cfg.signal_ml_sell_threshold,
    )
    bt_cfg = BacktestConfig(initial_capital=cfg.seed_usd, slippage_bps=5.0, commission_usd=0.0)
    try:
        result = run_backtest(ml_cfg, bt_cfg, test_bars_by_symbol)
        benchmark = run_buy_and_hold_benchmark(bt_cfg, test_bars_by_symbol)
        m, b = result["metrics"], benchmark["metrics"]
        print(
            f"ML signal   total return: {m['total_return']:+.2%}  Sharpe: {m['sharpe']:.2f}  "
            f"MaxDD: {m['max_drawdown']:.2%}  Trades: {m['num_trades']}"
        )
        print(f"Buy & Hold  total return: {b['total_return']:+.2%}  Sharpe: {b['sharpe']:.2f}")
        if m["total_return"] > b["total_return"]:
            print("ML signal beat buy-and-hold on this test window.")
        else:
            print("ML signal LOST to buy-and-hold on this test window -- accuracy above baseline does not guarantee a trading edge.")
    except ValueError as e:
        print(f"Could not run the downstream backtest on the test window: {e}")


if __name__ == "__main__":
    main()
