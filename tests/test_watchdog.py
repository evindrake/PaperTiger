"""Unit tests for watchdog.py: it must trip (create the kill file) ONLY
when the heartbeat is stale, never when it's fresh -- and never during the
startup grace period even if there's no heartbeat yet."""

import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import watchdog
from watchdog import check_once


def write_state(path: str, written_at: datetime) -> None:
    Path(path).write_text(json.dumps({"written_at": written_at.isoformat()}), encoding="utf-8")


class TestWatchdog(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.state_path = str(Path(self.tmpdir) / "runtime_state.json")
        self.kill_path = str(Path(self.tmpdir) / "HALT")

    def test_fresh_heartbeat_does_not_trip(self):
        now = time = __import__("time").time()
        write_state(self.state_path, datetime.now(timezone.utc))
        tripped = check_once(
            self.state_path, self.kill_path,
            stale_after_sec=300.0, started_at=now - 1000, grace_period_sec=120.0, now=now,
        )
        self.assertFalse(tripped)
        self.assertFalse(Path(self.kill_path).exists())

    def test_stale_heartbeat_trips(self):
        now = __import__("time").time()
        old_ts = datetime.now(timezone.utc) - timedelta(seconds=600)
        write_state(self.state_path, old_ts)
        tripped = check_once(
            self.state_path, self.kill_path,
            stale_after_sec=300.0, started_at=now - 1000, grace_period_sec=120.0, now=now,
        )
        self.assertTrue(tripped)
        self.assertTrue(Path(self.kill_path).exists())

    def test_missing_state_file_within_grace_period_does_not_trip(self):
        now = __import__("time").time()
        # no state file written at all yet -- engine just started
        tripped = check_once(
            self.state_path, self.kill_path,
            stale_after_sec=300.0, started_at=now - 10, grace_period_sec=120.0, now=now,
        )
        self.assertFalse(tripped)
        self.assertFalse(Path(self.kill_path).exists())

    def test_missing_state_file_after_grace_period_trips(self):
        now = __import__("time").time()
        tripped = check_once(
            self.state_path, self.kill_path,
            stale_after_sec=300.0, started_at=now - 1000, grace_period_sec=120.0, now=now,
        )
        self.assertTrue(tripped)
        self.assertTrue(Path(self.kill_path).exists())

    def test_already_triggered_kill_file_is_left_alone(self):
        now = __import__("time").time()
        Path(self.kill_path).write_text("HALT\noperator-triggered\n", encoding="utf-8")
        old_ts = datetime.now(timezone.utc) - timedelta(seconds=600)
        write_state(self.state_path, old_ts)
        tripped = check_once(
            self.state_path, self.kill_path,
            stale_after_sec=300.0, started_at=now - 1000, grace_period_sec=120.0, now=now,
        )
        self.assertFalse(tripped)  # not "newly" tripped -- it was already there
        contents = Path(self.kill_path).read_text(encoding="utf-8")
        self.assertIn("operator-triggered", contents)  # left untouched


class TestRunForeverNotifies(unittest.TestCase):
    def test_notifies_on_trip_when_cfg_given(self):
        tmpdir = tempfile.mkdtemp()
        state_path = str(Path(tmpdir) / "runtime_state.json")   # never written -- missing heartbeat
        kill_path = str(Path(tmpdir) / "HALT")
        notif_path = str(Path(tmpdir) / "notifications.json")
        cfg = SimpleNamespace(
            notify_smtp_host="", notify_to=(), notify_local_file_path=notif_path,
        )

        def fake_sleep(_secs):
            raise KeyboardInterrupt()  # stop the infinite loop after one iteration

        with patch("watchdog.time.sleep", side_effect=fake_sleep):
            with self.assertRaises(KeyboardInterrupt):
                watchdog.run_forever(
                    state_path, kill_path, stale_after_sec=300.0,
                    grace_period_sec=0.0, poll_interval_sec=0.01, cfg=cfg,
                )

        self.assertTrue(Path(kill_path).exists())
        data = json.loads(Path(notif_path).read_text(encoding="utf-8"))
        self.assertEqual(len(data), 1)
        self.assertIn("stale heartbeat", data[0]["subject"])


if __name__ == "__main__":
    unittest.main()
