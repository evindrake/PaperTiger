#!/usr/bin/env bash
# uninstall_services.sh -- removes everything install_services.sh set up:
# the systemd --user services/timers for PaperTiger's watchdog, dashboard,
# engine, daily retrain, self-test, and desktop notifier.
#
# This only removes the SERVICE REGISTRATIONS -- it does not touch any
# project files, results, or your .env. Your Alpaca account and any open
# paper positions are entirely unaffected; if you want to also flatten open
# positions, use the dashboard's kill switch (FLATTEN) before or after
# running this.

set -uo pipefail

UNIT_DIR="$HOME/.config/systemd/user"

for name in papertiger-watchdog papertiger-dashboard papertiger-engine papertiger-notifier; do
    echo "Stopping and removing ${name}.service..."
    systemctl --user disable --now "${name}.service" >/dev/null 2>&1 || true
    rm -f "$UNIT_DIR/${name}.service"
done

for name in papertiger-dailyretrain papertiger-selftest; do
    echo "Removing ${name}.timer / ${name}.service..."
    systemctl --user disable --now "${name}.timer" >/dev/null 2>&1 || true
    rm -f "$UNIT_DIR/${name}.timer" "$UNIT_DIR/${name}.service"
done

systemctl --user daemon-reload 2>/dev/null || true

echo "Done. Your Alpaca account and any open paper positions are unaffected --"
echo "use the dashboard's kill switch (FLATTEN) first if you also want to liquidate."
