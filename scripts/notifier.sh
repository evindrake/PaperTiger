#!/usr/bin/env bash
# notifier.sh -- Linux/macOS equivalent of tray_notifier.ps1. Watches
# notifications.json (written by notify.py whenever engine.py or watchdog.py
# has something worth knowing about: a circuit breaker trip, a kill-switch
# trigger, a stale-heartbeat watchdog intervention) and pops a desktop
# notification for each new entry.
#
# Uses whatever's actually available on this OS: `notify-send` on Linux
# (part of libnotify; install via your distro's package manager if missing,
# e.g. `apt install libnotify-bin`), `osascript` on macOS (built in, no
# install needed). Falls back to printing to stdout if neither exists, so
# nothing is silently lost -- check this script's own log file in that case.
#
# This script never reads .env, never talks to Alpaca, and cannot trade or
# touch the kill file -- its only capability is displaying a notification
# someone else (notify.py) already decided was worth surfacing.

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NOTIFICATIONS_FILE="$PROJECT_DIR/notifications.json"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
PYTHON="$VENV_PYTHON"
if [ ! -x "$PYTHON" ]; then
    PYTHON="python3"  # fall back to system python3 if the venv isn't set up yet
fi

osa_escape() {
    printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

notify() {
    local subject="$1" body="$2"
    if command -v notify-send >/dev/null 2>&1; then
        notify-send "PaperTiger: $subject" "$body"
    elif command -v osascript >/dev/null 2>&1; then
        local esc_body esc_subject
        esc_body=$(osa_escape "$body")
        esc_subject=$(osa_escape "$subject")
        osascript -e "display notification \"$esc_body\" with title \"PaperTiger\" subtitle \"$esc_subject\""
    else
        echo "[PaperTiger] $subject: $body"
    fi
}

# Don't replay the entire backlog on startup -- only show notifications that
# arrive AFTER this script starts. Seed last_seen_ts from whatever's already
# in the file (if anything). Paths/timestamps are passed as argv (not
# interpolated into the Python source) to sidestep any quoting issues.
last_seen_ts=$("$PYTHON" -c '
import json, sys
path = sys.argv[1]
try:
    data = json.load(open(path))
    print(data[-1]["ts"] if data else "")
except Exception:
    print("")
' "$NOTIFICATIONS_FILE")

while true; do
    if [ -f "$NOTIFICATIONS_FILE" ]; then
        new_entries=$("$PYTHON" -c '
import json, sys
path, last = sys.argv[1], sys.argv[2]
try:
    data = json.load(open(path))
except Exception:
    data = []
for n in data:
    ts = str(n.get("ts", ""))
    if not last or ts > last:
        subject = str(n.get("subject", "")).replace("\t", " ")
        body = str(n.get("body", "")).replace("\t", " ").replace("\n", " ")
        print(ts + "\t" + subject + "\t" + body)
' "$NOTIFICATIONS_FILE" "$last_seen_ts")

        if [ -n "$new_entries" ]; then
            while IFS=$'\t' read -r ts subject body; do
                [ -z "$ts" ] && continue
                notify "$subject" "$body"
                last_seen_ts="$ts"
            done <<< "$new_entries"
        fi
    fi
    sleep 10
done
