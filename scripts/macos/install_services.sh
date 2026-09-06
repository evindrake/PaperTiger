#!/usr/bin/env bash
# install_services.sh -- registers PaperTiger's background processes as
# launchd LaunchAgents, plus scheduled agents for the daily research
# retrain, weekly tactical universe refresh, and periodic self-test.
# macOS only.
#
# No sudo needed -- everything here is installed as a per-user LaunchAgent
# under ~/Library/LaunchAgents, so it has no more privilege than you do and
# runs only while you're logged in (same tradeoff as Windows' Scheduled
# Tasks for the tray notifier, and similar to systemd --user on Linux).
#
# NOTE: this script was written and reviewed carefully but has not been run
# against a real Mac as part of building PaperTiger (developed on Windows)
# -- read it before running it, and please open an issue if something
# doesn't work as described.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
AGENT_DIR="$HOME/Library/LaunchAgents"
LOG_DIR="$PROJECT_DIR/logs"

if [ ! -x "$VENV_PYTHON" ]; then
    echo "venv python not found at $VENV_PYTHON -- create the venv and install requirements first (see README.md)." >&2
    exit 1
fi

mkdir -p "$AGENT_DIR" "$LOG_DIR"

load_agent() {
    local label="$1" plist="$AGENT_DIR/${label}.plist"
    launchctl unload "$plist" >/dev/null 2>&1 || true
    launchctl load -w "$plist"
}

# -- Three persistent processes: watchdog, dashboard, engine --
install_persistent_agent() {
    local label="$1" script="$2" description="$3"
    cat > "$AGENT_DIR/${label}.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>$label</string>
    <key>ProgramArguments</key>
    <array>
        <string>$VENV_PYTHON</string>
        <string>$PROJECT_DIR/$script</string>
    </array>
    <key>WorkingDirectory</key><string>$PROJECT_DIR</string>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>StandardOutPath</key><string>$LOG_DIR/${label}.out.log</string>
    <key>StandardErrorPath</key><string>$LOG_DIR/${label}.err.log</string>
</dict>
</plist>
EOF
    load_agent "$label"
    echo "Installed and started $label ($description)"
}

install_persistent_agent "com.papertiger.watchdog" "watchdog.py" \
    "watchdog -- its ONLY power is creating the HALT kill file if the engine hangs. Never trades."
install_persistent_agent "com.papertiger.dashboard" "dashboard.py" \
    "status page at http://127.0.0.1:8787 with a kill-switch control and a Config tab for live risk-profile overrides. Cannot place a trade or touch the symbol whitelist."
install_persistent_agent "com.papertiger.engine" "run.py" \
    "the live (paper by default) trading loop. Refuses to trade real money unless ALPACA_PAPER=false AND I_UNDERSTAND_THIS_IS_REAL_MONEY=yes are both set in .env."

# -- Daily research retrain: fires once a day at 6am, not a persistent process --
cat > "$AGENT_DIR/com.papertiger.dailyretrain.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>com.papertiger.dailyretrain</string>
    <key>ProgramArguments</key>
    <array>
        <string>$PROJECT_DIR/scripts/daily_retrain.sh</string>
    </array>
    <key>WorkingDirectory</key><string>$PROJECT_DIR</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key><integer>6</integer>
        <key>Minute</key><integer>0</integer>
    </dict>
    <key>StandardOutPath</key><string>$LOG_DIR/com.papertiger.dailyretrain.out.log</string>
    <key>StandardErrorPath</key><string>$LOG_DIR/com.papertiger.dailyretrain.err.log</string>
</dict>
</plist>
EOF
load_agent "com.papertiger.dailyretrain"
echo "Installed com.papertiger.dailyretrain (daily at 6:00 AM)."

# -- Weekly tactical universe refresh: re-ranks the dynamic satellite pool
#    by liquidity (see scripts/refresh_tactical_universe.py) -- purely
#    additive on top of the static core SYMBOLS whitelist, never touches
#    core.py's bootstrap sizing (mirrors install_services.ps1's
#    PaperTiger-TacticalUniverseRefresh). --
cat > "$AGENT_DIR/com.papertiger.universerefresh.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>com.papertiger.universerefresh</string>
    <key>ProgramArguments</key>
    <array>
        <string>$VENV_PYTHON</string>
        <string>$PROJECT_DIR/scripts/refresh_tactical_universe.py</string>
    </array>
    <key>WorkingDirectory</key><string>$PROJECT_DIR</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Weekday</key><integer>0</integer>
        <key>Hour</key><integer>5</integer>
        <key>Minute</key><integer>0</integer>
    </dict>
    <key>StandardOutPath</key><string>$LOG_DIR/com.papertiger.universerefresh.out.log</string>
    <key>StandardErrorPath</key><string>$LOG_DIR/com.papertiger.universerefresh.err.log</string>
</dict>
</plist>
EOF
load_agent "com.papertiger.universerefresh"
echo "Installed com.papertiger.universerefresh (weekly, Sunday at 5:00 AM)."

# -- Self-test: runs at login and every 4 hours thereafter --
cat > "$AGENT_DIR/com.papertiger.selftest.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>com.papertiger.selftest</string>
    <key>ProgramArguments</key>
    <array>
        <string>$VENV_PYTHON</string>
        <string>$PROJECT_DIR/selftest.py</string>
    </array>
    <key>WorkingDirectory</key><string>$PROJECT_DIR</string>
    <key>RunAtLoad</key><true/>
    <key>StartInterval</key><integer>14400</integer>
    <key>StandardOutPath</key><string>$LOG_DIR/com.papertiger.selftest.out.log</string>
    <key>StandardErrorPath</key><string>$LOG_DIR/com.papertiger.selftest.err.log</string>
</dict>
</plist>
EOF
load_agent "com.papertiger.selftest"
echo "Installed com.papertiger.selftest (at login, then every 4 hours)."

echo ""
echo "Running the self-test once now..."
"$VENV_PYTHON" "$PROJECT_DIR/selftest.py" || true

# -- Desktop notifier: shows a notification banner for each new entry in
#    notifications.json while you're logged in (see scripts/notifier.sh) --
cat > "$AGENT_DIR/com.papertiger.notifier.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>com.papertiger.notifier</string>
    <key>ProgramArguments</key>
    <array>
        <string>$PROJECT_DIR/scripts/notifier.sh</string>
    </array>
    <key>WorkingDirectory</key><string>$PROJECT_DIR</string>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>StandardOutPath</key><string>$LOG_DIR/com.papertiger.notifier.out.log</string>
    <key>StandardErrorPath</key><string>$LOG_DIR/com.papertiger.notifier.err.log</string>
</dict>
</plist>
EOF
load_agent "com.papertiger.notifier"
echo "Installed com.papertiger.notifier (desktop notification popups)."

echo ""
echo "All done. Check status with:"
echo "  launchctl list | grep papertiger"
echo ""
echo "Stop everything at any time with the dashboard's kill switch, or"
echo "'launchctl stop com.papertiger.engine'. To fully remove, run"
echo "scripts/macos/uninstall_services.sh."
echo ""
echo "Note: LaunchAgents (unlike Windows services) only run while you're"
echo "logged in -- if this Mac needs to keep trading while logged out, these"
echo "would need to be installed as system-level LaunchDaemons instead"
echo "(different setup, not covered by this script)."
