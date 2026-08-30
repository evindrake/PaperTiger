"""Unit tests for trade_log.py."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trade_log import append_trade


class TestAppendTrade(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = str(Path(self.tmpdir) / "trade_history.jsonl")

    def test_appends_one_json_line(self):
        append_trade(
            self.path, source="tactical", symbol="SPY", side="buy",
            reason="sma_crossover buy signal", notional_usd=25.0, limit_price=450.5,
            order_id="o1", client_order_id="pt-SPY-buy-2026-08-23",
        )
        lines = Path(self.path).read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)
        record = json.loads(lines[0])
        self.assertEqual(record["symbol"], "SPY")
        self.assertEqual(record["side"], "buy")
        self.assertEqual(record["source"], "tactical")
        self.assertEqual(record["notional_usd"], 25.0)

    def test_multiple_appends_accumulate(self):
        for i in range(5):
            append_trade(self.path, source="tactical", symbol="SPY", side="buy", reason=f"r{i}")
        lines = Path(self.path).read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 5)

    def test_core_bootstrap_source_recorded(self):
        append_trade(
            self.path, source="core_bootstrap", symbol="QQQ", side="buy",
            reason="core-satellite bootstrap", notional_usd=50.0,
        )
        record = json.loads(Path(self.path).read_text(encoding="utf-8").strip())
        self.assertEqual(record["source"], "core_bootstrap")

    def test_unwritable_path_does_not_raise(self):
        bad_path = str(Path(self.tmpdir) / "nonexistent_dir" / "trade_history.jsonl")
        append_trade(bad_path, source="tactical", symbol="SPY", side="buy", reason="test")  # must not raise


if __name__ == "__main__":
    unittest.main()
