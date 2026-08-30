#!/usr/bin/env bash
# daily_retrain.sh -- Linux/macOS equivalent of daily_retrain.ps1. Reruns
# backtest.py, walkforward.py, and train_ml_signal.py against fresh Alpaca
# historical data, refreshing backtest_results.json / walkforward_results.json
# / ml_model.joblib for the dashboard to display.
#
# This is the "keep learning" half of PaperTiger's automation. run.py itself
# just executes whatever signal is currently configured in .env -- it never
# experiments on its own. This script is what actually keeps searching for a
# configuration that might clear the "beats buy-and-hold AND survives
# walk-forward" bar. It runs once and exits; the daily timer/agent installed
# by scripts/linux/install_services.sh or scripts/macos/install_services.sh
# is what runs it every day.
#
# None of this places any order -- backtest.py/walkforward.py/
# train_ml_signal.py are all offline analysis tools with no broker
# connection for trading, only for pulling historical bars.
#
# Unlike daily_retrain.ps1, this script does NOT prune old logs -- NSSM's
# size-based log rotation (what that cleanup step exists for) has no direct
# Linux/macOS equivalent here. If logs/daily_retrain.log grows large over
# time, consider standard tools like `logrotate` (Linux) instead.

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/daily_retrain.log"

if [ ! -x "$VENV_PYTHON" ]; then
    echo "venv python not found at $VENV_PYTHON -- create the venv and install requirements first (see README.md)." >&2
    exit 1
fi

TODAY=$(date +%Y-%m-%d)
# GNU date (Linux) uses -d; BSD date (macOS) uses -v. Try GNU syntax first,
# fall back to BSD syntax if that fails.
START=$(date -d "-4 years" +%Y-%m-%d 2>/dev/null || date -v-4y +%Y-%m-%d)

{
    echo "=== PaperTiger daily research run: $(date -Is 2>/dev/null || date) ==="
} >> "$LOG_FILE"

echo "Running backtest.py ($START to $TODAY)..."
"$VENV_PYTHON" backtest.py --source alpaca --start "$START" --end "$TODAY" --out backtest_results.json 2>&1 | tee -a "$LOG_FILE"

echo "Running walkforward.py ($START to $TODAY)..."
"$VENV_PYTHON" walkforward.py --source alpaca --start "$START" --end "$TODAY" --out walkforward_results.json 2>&1 | tee -a "$LOG_FILE"

echo "Running train_ml_signal.py ($START to $TODAY)..."
"$VENV_PYTHON" train_ml_signal.py --source alpaca --start "$START" --end "$TODAY" --model-out ml_model.joblib 2>&1 | tee -a "$LOG_FILE"

{
    echo "=== done: $(date -Is 2>/dev/null || date) ==="
} >> "$LOG_FILE"
echo "Done. Full log: $LOG_FILE"
