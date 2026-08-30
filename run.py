"""
run.py -- entrypoint. Wires config -> broker -> strategy -> engine and
starts the live (paper, by default) trading loop.

This does NOT run preflight.py or backtest.py/walkforward.py for you --
the README's "typical session" runbook is: preflight -> backtest ->
walkforward -> (only if those hold up) watchdog + dashboard + run. Run
those separately and read their output before you get here.

cfg.guard_live() is called before anything else touches the network. It is
already invoked once inside config.load_config() when alpaca_paper is
False, but we call it again explicitly here so this file's own safety
story is readable without having to go trace through config.py -- belt and
suspenders on the single most consequential check in the project.
"""

from __future__ import annotations

import sys

from broker import Broker
from config import load_config
from engine import Engine


def main() -> int:
    cfg = load_config()
    cfg.guard_live()  # refuses to proceed live unless both safety flags are set

    mode = "PAPER" if cfg.alpaca_paper else "LIVE"
    print(f"[run] starting PaperTiger engine in {mode} mode")
    print(f"[run] symbols: {', '.join(cfg.symbols)}")
    print(f"[run] target trade size: ${cfg.target_trade_usd:.2f} | kill file: {cfg.kill_file_path}")

    broker = Broker(cfg)
    engine = Engine(cfg, broker=broker)

    try:
        engine.run_forever()
    except KeyboardInterrupt:
        print("\n[run] stopped by operator (Ctrl+C)")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
