#!/usr/bin/env bash
# install_services.sh -- registers PaperTiger's background processes as
# systemd --user services, plus timers for the daily research retrain and
# periodic self-test. Linux only.
#
# No sudo/root needed -- everything here is installed under YOUR OWN user
# account via `systemctl --user`, so it has no more privilege than you do
# and never touches system-wide state.
#
# NOTE: this script was written and reviewed carefully but has not been run
# against a real Linux machine as part of building PaperTiger (developed on
# Windows) -- read it before running it, and please open an issue if
# something doesn't work as described.
#
# For these services to keep running after you log out (e.g. a headless
# server, or closing an SSH session), enable lingering ONCE:
#   loginctl enable-linger "$USER"
# Without that, systemd --user services stop when your last login session ends.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
UNIT_DIR="$HOME/.config/systemd/user"
LOG_DIR="$PROJECT_DIR/logs"

if [ ! -x "$VENV_PYTHON" ]; then
    echo "venv python not found at $VENV_PYTHON -- create the venv and install requirements first (see README.md)." >&2
    exit 1
fi
if ! command -v systemctl >/dev/null 2>&1; then
    echo "systemctl not found -- this script requires systemd (most modern Linux distros have it)." >&2
    exit 1
fi

mkdir -p "$UNIT_DIR" "$LOG_DIR"

install_persistent_service() {
    local name="$1" script="$2" description="$3"
    cat > "$UNIT_DIR/${name}.service" <<EOF
[Unit]
Description=$description
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$PROJECT_DIR
ExecStart=$VENV_PYTHON $PROJECT_DIR/$script
Restart=on-failure
RestartSec=5
StandardOutput=append:$LOG_DIR/${name}.out.log
StandardError=append:$LOG_DIR/${name}.err.log

[Install]
WantedBy=default.target
EOF
    systemctl --user daemon-reload
    systemctl --user enable --now "${name}.service"
    echo "Installed and started ${name}.service"
}

install_persistent_service "papertiger-watchdog" "watchdog.py" \
    "PaperTiger: watches runtime_state.json's heartbeat; its ONLY power is creating the HALT kill file if the engine hangs. Never trades."

install_persistent_service "papertiger-dashboard" "dashboard.py" \
    "PaperTiger: status page at http://127.0.0.1:8787 with a kill-switch control. Cannot place a trade."

install_persistent_service "papertiger-engine" "run.py" \
    "PaperTiger: the live (paper by default) trading loop. Refuses to trade real money unless ALPACA_PAPER=false AND I_UNDERSTAND_THIS_IS_REAL_MONEY=yes are both set in .env."

# -- Daily research retrain: a oneshot service triggered by a timer, not a
#    long-running process (mirrors install_services.ps1's PaperTiger-DailyResearch). --
cat > "$UNIT_DIR/papertiger-dailyretrain.service" <<EOF
[Unit]
Description=PaperTiger: daily rerun of backtest.py + walkforward.py + train_ml_signal.py against fresh data.

[Service]
Type=oneshot
WorkingDirectory=$PROJECT_DIR
ExecStart=$PROJECT_DIR/scripts/daily_retrain.sh
EOF

cat > "$UNIT_DIR/papertiger-dailyretrain.timer" <<EOF
[Unit]
Description=Run papertiger-dailyretrain.service daily at 6am

[Timer]
OnCalendar=*-*-* 06:00:00
Persistent=true

[Install]
WantedBy=timers.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now papertiger-dailyretrain.timer
echo "Installed papertiger-dailyretrain.timer (daily at 6:00 AM)."

# -- Self-test: runs the full unit test suite shortly after boot/login and
#    every 4 hours thereafter, to catch environment drift in an unattended
#    deployment (see selftest.py's module docstring). --
cat > "$UNIT_DIR/papertiger-selftest.service" <<EOF
[Unit]
Description=PaperTiger: runs the full unit test suite; writes selftest_results.json for the dashboard's Status tab.

[Service]
Type=oneshot
WorkingDirectory=$PROJECT_DIR
ExecStart=$VENV_PYTHON $PROJECT_DIR/selftest.py
EOF

cat > "$UNIT_DIR/papertiger-selftest.timer" <<EOF
[Unit]
Description=Run papertiger-selftest.service at boot and every 4 hours

[Timer]
OnBootSec=1min
OnUnitActiveSec=4h
Persistent=true

[Install]
WantedBy=timers.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now papertiger-selftest.timer
echo "Installed papertiger-selftest.timer (shortly after boot/login, then every 4 hours)."

echo ""
echo "Running the self-test once now..."
"$VENV_PYTHON" "$PROJECT_DIR/selftest.py" || true

# -- Desktop notifier: shows a desktop popup for each new entry in
#    notifications.json while you're logged in (see scripts/notifier.sh). --
cat > "$UNIT_DIR/papertiger-notifier.service" <<EOF
[Unit]
Description=PaperTiger: desktop notification popup for kill-switch/circuit-breaker/watchdog events.
After=graphical-session.target

[Service]
Type=simple
WorkingDirectory=$PROJECT_DIR
ExecStart=$PROJECT_DIR/scripts/notifier.sh
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
if command -v notify-send >/dev/null 2>&1; then
    systemctl --user enable --now papertiger-notifier.service
    echo "Installed and started papertiger-notifier.service."
else
    echo "notify-send not found -- skipping desktop notifier for now (install libnotify, e.g."
    echo "'apt install libnotify-bin' / 'dnf install libnotify', then rerun this script to add"
    echo "it. Email/SMS notifications via NOTIFY_TO in .env still work regardless)."
fi

echo ""
echo "All done. Check status with:"
echo "  systemctl --user status 'papertiger-*'"
echo ""
echo "Stop everything at any time with the dashboard's kill switch, or"
echo "'systemctl --user stop papertiger-engine'. To fully remove, run"
echo "scripts/linux/uninstall_services.sh."
echo ""
echo "Reminder: if this machine won't stay logged in as you (e.g. a headless"
echo "server), run 'loginctl enable-linger \"\$USER\"' once so these keep running"
echo "after you log out."
