"""Unit tests for equity_history.py: the self-pruning JSONL log behind the
Live tab's "last 30 days" performance chart."""

import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from equity_history import append_snapshot, read_recent


class TestAppendSnapshot(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = str(Path(self.tmpdir) / "equity_history.jsonl")

    def test_appends_one_json_line(self):
        append_snapshot(self.path, equity=210.0, cash=10.0, positions_value={"SPY": 100.0, "QQQ": 100.0})
        lines = Path(self.path).read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)
        record = json.loads(lines[0])
        self.assertEqual(record["equity"], 210.0)
        self.assertEqual(record["cash"], 10.0)
        self.assertEqual(record["positions_value"], 200.0)
        self.assertEqual(record["positions"], {"SPY": 100.0, "QQQ": 100.0})

    def test_multiple_appends_accumulate(self):
        for i in range(5):
            append_snapshot(self.path, equity=200.0 + i, cash=10.0, positions_value={})
        lines = Path(self.path).read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 5)

    def test_empty_positions_value_gives_zero_total(self):
        append_snapshot(self.path, equity=200.0, cash=200.0, positions_value={})
        record = json.loads(Path(self.path).read_text(encoding="utf-8").strip())
        self.assertEqual(record["positions_value"], 0.0)

    def test_entries_older_than_retention_are_pruned(self):
        old_ts = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
        Path(self.path).write_text(
            json.dumps({"ts": old_ts, "equity": 100.0, "cash": 100.0, "positions_value": 0.0, "positions": {}}) + "\n",
            encoding="utf-8",
        )
        append_snapshot(self.path, equity=250.0, cash=50.0, positions_value={"SPY": 200.0}, retention_days=35)
        lines = Path(self.path).read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)  # the 40-day-old entry was pruned, only the new one remains
        record = json.loads(lines[0])
        self.assertEqual(record["equity"], 250.0)

    def test_unwritable_path_does_not_raise(self):
        bad_path = str(Path(self.tmpdir) / "nonexistent_dir" / "equity_history.jsonl")
        append_snapshot(bad_path, equity=200.0, cash=200.0, positions_value={})  # must not raise


class TestReadRecent(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = str(Path(self.tmpdir) / "equity_history.jsonl")

    def test_missing_file_gives_empty_list(self):
        self.assertEqual(read_recent(self.path), [])

    def test_returns_entries_within_window_sorted_oldest_first(self):
        now = datetime.now(timezone.utc)
        lines = []
        for days_ago, equity in [(2, 220.0), (1, 230.0), (0, 240.0)]:
            ts = (now - timedelta(days=days_ago)).isoformat()
            lines.append(json.dumps({"ts": ts, "equity": equity, "cash": 0.0, "positions_value": equity, "positions": {}}))
        Path(self.path).write_text("\n".join(lines) + "\n", encoding="utf-8")

        result = read_recent(self.path, days=30)
        self.assertEqual([e["equity"] for e in result], [220.0, 230.0, 240.0])

    def test_excludes_entries_outside_the_window(self):
        now = datetime.now(timezone.utc)
        old_ts = (now - timedelta(days=60)).isoformat()
        recent_ts = (now - timedelta(days=1)).isoformat()
        lines = [
            json.dumps({"ts": old_ts, "equity": 100.0, "cash": 0.0, "positions_value": 100.0, "positions": {}}),
            json.dumps({"ts": recent_ts, "equity": 200.0, "cash": 0.0, "positions_value": 200.0, "positions": {}}),
        ]
        Path(self.path).write_text("\n".join(lines) + "\n", encoding="utf-8")

        result = read_recent(self.path, days=30)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["equity"], 200.0)

    def test_corrupted_line_is_skipped_not_fatal(self):
        good_ts = datetime.now(timezone.utc).isoformat()
        content = "not valid json\n" + json.dumps({"ts": good_ts, "equity": 200.0, "cash": 0.0, "positions_value": 200.0, "positions": {}}) + "\n"
        Path(self.path).write_text(content, encoding="utf-8")

        result = read_recent(self.path, days=30)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["equity"], 200.0)


if __name__ == "__main__":
    unittest.main()
