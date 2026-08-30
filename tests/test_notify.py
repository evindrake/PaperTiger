"""Unit tests for notify.py -- SMTP is mocked out entirely, no real network
calls or emails are ever sent by these tests."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import notify


def make_cfg(tmpdir=None, **overrides):
    base = dict(
        notify_smtp_host="", notify_smtp_port=587, notify_smtp_username="",
        notify_smtp_password="", notify_from_email="", notify_to=(),
        notify_local_file_path=str(Path(tmpdir) / "notifications.json") if tmpdir else "notifications.json",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestSendNotification(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_disabled_when_not_configured_returns_false(self):
        cfg = make_cfg(self.tmpdir)  # no host, no recipients
        sent = notify.send_notification(cfg, "test subject", "test body")
        self.assertFalse(sent)

    def test_disabled_still_writes_local_notification(self):
        cfg = make_cfg(self.tmpdir)
        notify.send_notification(cfg, "halted", "circuit breaker tripped")
        data = json.loads(Path(cfg.notify_local_file_path).read_text(encoding="utf-8"))
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["subject"], "halted")
        self.assertEqual(data[0]["body"], "circuit breaker tripped")

    @patch("notify.smtplib.SMTP")
    def test_configured_sends_email_and_returns_true(self, mock_smtp_cls):
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__.return_value = mock_server

        cfg = make_cfg(
            self.tmpdir,
            notify_smtp_host="smtp.example.com", notify_smtp_port=587,
            notify_smtp_username="me@example.com", notify_smtp_password="app-password",
            notify_from_email="me@example.com", notify_to=("me@example.com", "5551234567@vtext.com"),
        )
        sent = notify.send_notification(cfg, "HALT triggered", "the engine halted itself")

        self.assertTrue(sent)
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with("me@example.com", "app-password")
        mock_server.sendmail.assert_called_once()
        args = mock_server.sendmail.call_args[0]
        self.assertEqual(args[0], "me@example.com")
        self.assertIn("5551234567@vtext.com", args[1])

    @patch("notify.smtplib.SMTP", side_effect=OSError("connection refused"))
    def test_smtp_failure_is_swallowed_not_raised(self, mock_smtp_cls):
        cfg = make_cfg(
            self.tmpdir,
            notify_smtp_host="smtp.example.com", notify_to=("me@example.com",),
        )
        sent = notify.send_notification(cfg, "subject", "body")  # must not raise
        self.assertFalse(sent)

    def test_local_notifications_capped_and_most_recent_last(self):
        cfg = make_cfg(self.tmpdir)
        for i in range(60):
            notify.send_notification(cfg, f"subject-{i}", f"body-{i}")
        data = json.loads(Path(cfg.notify_local_file_path).read_text(encoding="utf-8"))
        self.assertEqual(len(data), notify._MAX_LOCAL_NOTIFICATIONS)
        self.assertEqual(data[-1]["subject"], "subject-59")


if __name__ == "__main__":
    unittest.main()
