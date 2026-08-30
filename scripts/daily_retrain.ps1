<#
daily_retrain.ps1 -- reruns backtest.py, walkforward.py, and
train_ml_signal.py against fresh Alpaca historical data, refreshing
backtest_results.json / walkforward_results.json / ml_model.joblib for the
dashboard to display. Also cleans up old rotated NSSM log files (see the
bottom of this script) -- NSSM rotates at 5MB but never deletes the old
copies, so without this they'd accumulate indefinitely.

This is the "keep learning" half of PaperTiger's automation. run.py itself
just executes whatever signal is currently configured in .env -- it never
experiments on its own. This script is what actually keeps searching for a
configuration that might clear the "beats buy-and-hold AND survives
walk-forward" bar. It runs once and exits; PaperTiger-DailyResearch (the
Scheduled Task install_services.ps1 registers) is what runs it daily.

None of this places any order -- backtest.py/walkforward.py/
train_ml_signal.py are all offline analysis tools with no broker
connection for trading, only for pulling historical bars.
#>

$ErrorActionPreference = "Continue"  # one step failing shouldn't cancel the others

$ProjectDir = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectDir
$VenvPython = Join-Path $ProjectDir ".venv\Scripts\python.exe"
$LogDir = Join-Path $ProjectDir "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogFile = Join-Path $LogDir "daily_retrain.log"

$Today = Get-Date -Format "yyyy-MM-dd"
# 4 years of daily bars is plenty for this small a symbol whitelist without
# needing anything beyond the free IEX data feed.
$Start = (Get-Date).AddYears(-4).ToString("yyyy-MM-dd")

"=== PaperTiger daily research run: $(Get-Date -Format s) ===" | Add-Content -Path $LogFile

Write-Host "Running backtest.py ($Start to $Today)..."
& $VenvPython backtest.py --source alpaca --start $Start --end $Today --out backtest_results.json 2>&1 |
    Tee-Object -FilePath $LogFile -Append

Write-Host "Running walkforward.py ($Start to $Today)..."
& $VenvPython walkforward.py --source alpaca --start $Start --end $Today --out walkforward_results.json 2>&1 |
    Tee-Object -FilePath $LogFile -Append

Write-Host "Running train_ml_signal.py ($Start to $Today)..."
& $VenvPython train_ml_signal.py --source alpaca --start $Start --end $Today --model-out ml_model.joblib 2>&1 |
    Tee-Object -FilePath $LogFile -Append

"=== done: $(Get-Date -Format s) ===" | Add-Content -Path $LogFile
Write-Host "Done. Full log: $LogFile"

# -- Log retention: delete rotated NSSM logs older than 30 days. --
# NSSM's AppRotateFiles/AppRotateBytes settings (see install_services.ps1)
# rename the current log to <name>-<timestamp>.log when it hits 5MB and
# start a fresh one, but never delete the old copies themselves. The live
# PaperTiger-*.out.log / .err.log files (no timestamp in the name) are
# NEVER touched here -- only their timestamped, already-rotated-out copies.
$RetentionDays = 30
$CutoffDate = (Get-Date).AddDays(-$RetentionDays)
$RotatedLogs = Get-ChildItem -Path $LogDir -Filter "*-*.log" -ErrorAction SilentlyContinue |
    Where-Object { $_.LastWriteTime -lt $CutoffDate }
if ($RotatedLogs) {
    Write-Host "Removing $($RotatedLogs.Count) rotated log file(s) older than $RetentionDays days..."
    $RotatedLogs | Remove-Item -Force -ErrorAction SilentlyContinue
}
