"""
notify.py -- optional, best-effort email/SMS notifications for events that
matter when nobody's watching the dashboard: circuit-breaker trips,
kill-switch triggers, watchdog interventions.

FREE by construction, on purpose:
  - Email uses plain SMTP (stdlib smtplib) through your OWN email
    provider's outgoing mail server (e.g. Gmail's smtp.gmail.com with an
    "app password") -- no paid notification service, no third-party API key.
  - "Text message" delivery works through the same mechanism: every major
    US carrier has a free email-to-SMS gateway (e.g. 5551234567@vtext.com
    for Verizon, 5551234567@txt.att.net for AT&T, 5551234567@tmomail.net
    for T-Mobile). Add that address as just another entry in NOTIFY_TO and
    it arrives as a text message, at no cost, via the same email send.

Also ALWAYS writes a local notifications.json (rolling, most-recent-last)
regardless of whether SMTP is configured -- this is what
scripts/tray_notifier.ps1 polls to show a system-tray balloon while you're
at the PC, so the toast/tray notification works even with zero email setup.

This module NEVER raises into its caller: a notification failure (bad
credentials, no network, SMTP server down) is printed and swallowed, never
allowed to interrupt a trading decision or crash the engine/watchdog. A
failed notification is a missed heads-up, not a reason to also break the
thing it was trying to tell you about.
"""

from __future__ import annotations

import json
import smtplib
import ssl
import sys
from datetime import datetime, timezone
from email.mime.text import MIMEText
from pathlib import Path
from typing import Sequence

NOTIFICATIONS_FILE = "notifications.json"
_MAX_LOCAL_NOTIFICATIONS = 50


def _append_local_notification(subject: str, body: str, local_file_path: str) -> None:
    """Best-effort local record for the tray notifier -- failures here are
    swallowed too, same policy as the rest of this module."""
    path = Path(local_file_path)
    try:
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    except (OSError, json.JSONDecodeError):
        existing = []
    existing.append({
        "ts": datetime.now(timezone.utc).isoformat(),
        "subject": subject,
        "body": body,
    })
    existing = existing[-_MAX_LOCAL_NOTIFICATIONS:]
    try:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        tmp.replace(path)
    except OSError as e:
        print(f"[notify] failed to write local notification file: {e}", file=sys.stderr)


def _send_email(
    host: str, port: int, username: str, password: str, from_addr: str, to_addrs: Sequence[str],
    subject: str, body: str,
) -> bool:
    msg = MIMEText(body)
    msg["Subject"] = f"[PaperTiger] {subject}"
    msg["From"] = from_addr
    msg["To"] = ", ".join(to_addrs)

    context = ssl.create_default_context()
    with smtplib.SMTP(host, port, timeout=10) as server:
        server.starttls(context=context)
        if username:
            server.login(username, password)
        server.sendmail(from_addr, list(to_addrs), msg.as_string())
    return True


def send_notification(cfg, subject: str, body: str) -> bool:
    """Best-effort email/SMS notification, plus always a local record for
    the tray notifier. Returns True if an email/SMS was actually sent,
    False if skipped (not configured) or failed -- never raises.

    `cfg` needs (all optional; missing host/to means "skip SMTP entirely"):
      notify_smtp_host, notify_smtp_port, notify_smtp_username,
      notify_smtp_password, notify_from_email, notify_to (tuple of strings),
      notify_local_file_path (where the local record goes; defaults to
      NOTIFICATIONS_FILE in the current directory, same as tray_notifier.ps1
      expects for a real deployment -- tests should override this to a
      tmpdir path so they don't write into the real project directory).
    """
    local_file_path = getattr(cfg, "notify_local_file_path", "") or NOTIFICATIONS_FILE
    _append_local_notification(subject, body, local_file_path)

    host = getattr(cfg, "notify_smtp_host", "") or ""
    to_addrs = getattr(cfg, "notify_to", ()) or ()
    if not host or not to_addrs:
        return False  # not configured -- local notification still recorded above

    port = getattr(cfg, "notify_smtp_port", 587) or 587
    username = getattr(cfg, "notify_smtp_username", "") or ""
    password = getattr(cfg, "notify_smtp_password", "") or ""
    from_addr = getattr(cfg, "notify_from_email", "") or username or (to_addrs[0] if to_addrs else "")

    try:
        _send_email(host, port, username, password, from_addr, to_addrs, subject, body)
        return True
    except Exception as e:
        # Never let a notification failure propagate -- print for the
        # service's own log (NSSM captures stderr) and move on.
        print(f"[notify] failed to send email/SMS notification: {e}", file=sys.stderr)
        return False
