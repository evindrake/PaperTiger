#!/usr/bin/env bash
# uninstall_services.sh -- removes everything install_services.sh set up:
# the LaunchAgents for PaperTiger's watchdog, dashboard, engine, daily
# retrain, self-test, and desktop notifier.
#
# This only removes the AGENT REGISTRATIONS -- it does not touch any
# project files, results, or your .env. Your Alpaca account and any open
# paper positions are entirely unaffected; if you want to also flatten open
# positions, use the dashboard's kill switch (FLATTEN) before or after
# running this.

set -uo pipefail

AGENT_DIR="$HOME/Library/LaunchAgents"

for label in com.papertiger.watchdog com.papertiger.dashboard com.papertiger.engine \
             com.papertiger.dailyretrain com.papertiger.selftest com.papertiger.notifier; do
    plist="$AGENT_DIR/${label}.plist"
    if [ -f "$plist" ]; then
        echo "Stopping and removing $label..."
        launchctl unload "$plist" >/dev/null 2>&1 || true
        rm -f "$plist"
    else
        echo "$label not installed -- skipping."
    fi
done

echo "Done. Your Alpaca account and any open paper positions are unaffected --"
echo "use the dashboard's kill switch (FLATTEN) first if you also want to liquidate."
