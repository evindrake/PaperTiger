"""
generate_synthetic_data.py -- writes fake daily OHLCV CSVs into data/ so you
can run backtest.py and walkforward.py without any network access or an
Alpaca account.

This is a LEARNING convenience, not a market simulator: prices are a plain
geometric random walk with a configurable annualized drift and volatility,
seeded for reproducibility. It exists purely to exercise the backtest
plumbing (loading, warmup, order sizing, caps, metrics, benchmark) end to
end -- it has no relationship to how any real security actually behaves,
and a strategy "working" on this data means nothing about real markets.

Usage:
    python generate_synthetic_data.py
    python generate_synthetic_data.py --days 750 --out-dir data

Writes one CSV per symbol in --symbols, each with columns:
    date,open,high,low,close,volume
oldest date first, one row per (simulated) trading day (weekends skipped,
US holidays are NOT modeled -- see LIMITATIONS in backtest.py).
"""

from __future__ import annotations

import argparse
import csv
import random
from datetime import date, timedelta
from pathlib import Path
from typing import List, Tuple

# (symbol -> (annual_drift, annual_volatility, starting_price)). Numbers are
# rough, illustrative ballparks for a "boring diversified ETF" -- not
# calibrated to any real historical data.
_DEFAULT_PROFILES = {
    "SPY": (0.08, 0.16, 450.0),
    "QQQ": (0.10, 0.22, 380.0),
    "VTI": (0.08, 0.16, 240.0),
    "IVV": (0.08, 0.16, 460.0),
}


def _trading_days(start: date, n: int) -> List[date]:
    """n weekday dates starting from `start` (inclusive), skipping
    Sat/Sun. No holiday calendar -- good enough for synthetic data."""
    days = []
    d = start
    while len(days) < n:
        if d.weekday() < 5:  # Mon-Fri
            days.append(d)
        d += timedelta(days=1)
    return days


def _simulate_ohlcv(
    n: int, annual_drift: float, annual_vol: float, start_price: float, seed: int
) -> List[Tuple[float, float, float, float, float]]:
    """Simple daily geometric-random-walk close series, with open/high/low
    derived by adding small intraday noise around the close-to-close move,
    and volume as plain uniform noise. Returns list of (open,high,low,close,volume)."""
    rng = random.Random(seed)
    trading_days_per_year = 252
    mu_daily = annual_drift / trading_days_per_year
    sigma_daily = annual_vol / (trading_days_per_year ** 0.5)

    bars = []
    prev_close = start_price
    for _ in range(n):
        today_open = prev_close * (1 + rng.gauss(0, sigma_daily * 0.25))
        shock = rng.gauss(mu_daily, sigma_daily)
        today_close = today_open * (1 + shock)
        today_close = max(today_close, 0.01)  # keep prices positive
        high = max(today_open, today_close) * (1 + abs(rng.gauss(0, sigma_daily * 0.3)))
        low = min(today_open, today_close) * (1 - abs(rng.gauss(0, sigma_daily * 0.3)))
        low = max(low, 0.01)
        volume = rng.uniform(1_000_000, 8_000_000)
        bars.append((today_open, high, low, today_close, volume))
        prev_close = today_close
    return bars


def write_csv(symbol: str, out_dir: Path, days: int, start_date: date, seed: int) -> Path:
    drift, vol, start_price = _DEFAULT_PROFILES.get(symbol, (0.07, 0.20, 100.0))
    dates = _trading_days(start_date, days)
    bars = _simulate_ohlcv(days, drift, vol, start_price, seed=seed + sum(map(ord, symbol)))

    out_path = out_dir / f"{symbol}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "open", "high", "low", "close", "volume"])
        for d, (o, h, l, c, v) in zip(dates, bars):
            writer.writerow([d.isoformat(), f"{o:.4f}", f"{h:.4f}", f"{l:.4f}", f"{c:.4f}", f"{v:.0f}"])
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default="SPY,QQQ,VTI,IVV", help="comma-separated symbol list")
    parser.add_argument("--days", type=int, default=756, help="number of simulated trading days (~3 years)")
    parser.add_argument("--start-date", default="2022-01-03", help="YYYY-MM-DD first simulated trading day")
    parser.add_argument("--out-dir", default="data", help="output directory")
    parser.add_argument("--seed", type=int, default=42, help="base random seed (reproducible output)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start_date = date.fromisoformat(args.start_date)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    for symbol in symbols:
        path = write_csv(symbol, out_dir, args.days, start_date, args.seed)
        print(f"wrote {args.days} synthetic daily bars for {symbol} -> {path}")


if __name__ == "__main__":
    main()
