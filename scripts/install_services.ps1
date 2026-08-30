<#
install_services.ps1 -- registers PaperTiger's background processes as
Windows services (via NSSM) plus a daily scheduled research task.

Run this ONCE, from an ELEVATED (Administrator) PowerShell prompt.

Installs three services, all named with a PaperTiger- prefix so they're
easy to find in services.msc / Task Manager / Get-Service:

  - PaperTiger-Watchdog   (watchdog.py)  -- its ONLY power is creating the
    HALT kill file if the engine's heartbeat goes stale. Never trades.
  - PaperTiger-Dashboard  (dashboard.py) -- read-only status page at
    http://127.0.0.1:8787 plus a kill-switch control. Cannot place a trade.
  - PaperTiger-Engine     (run.py)       -- the live (paper by default)
    trading loop.

All three are set to auto-start at boot and auto-restart a few seconds
after a crash. PaperTiger-Engine is safe to auto-start unattended because
config.py's guard_live() refuses to trade against a real-money account
unless BOTH ALPACA_PAPER=false AND I_UNDERSTAND_THIS_IS_REAL_MONEY=yes are
explicitly set in .env -- a service that auto-restarts after a crash just
re-runs the same command against whatever .env currently says. It cannot
flip itself from paper to live; only a human editing .env can do that, and
the moment they do, guard_live() re-checks before anything else happens.

Also registers a daily Scheduled Task, PaperTiger-DailyResearch, that
reruns backtest.py + walkforward.py + train_ml_signal.py against fresh
data every day (see daily_retrain.ps1) -- this is the part that actually
keeps "searching" for a better configuration; run.py itself just executes
whatever signal is currently configured.

Logs for each service go to logs\<ServiceName>.out.log / .err.log in the
project directory, rotated at 5MB.
#>

#Requires -RunAsAdministrator

$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectDir ".venv\Scripts\python.exe"
$LogDir = Join-Path $ProjectDir "logs"

function Resolve-NssmPath {
    $cmd = Get-Command nssm.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $found = Get-ChildItem -Path "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Filter "nssm.exe" -Recurse -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -like "*win64*" } | Select-Object -First 1
    if ($found) { return $found.FullName }
    throw "nssm.exe not found -- install it first: winget install NSSM.NSSM"
}

$NssmExe = Resolve-NssmPath

if (-not (Test-Path $VenvPython)) {
    throw "venv python not found at $VenvPython -- create the venv and install requirements first (see README.md)."
}
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Install-PtService {
    param(
        [string]$Name,
        [string]$ScriptFile,
        [string]$DisplayName,
        [string]$Description
    )

    if (Get-Service -Name $Name -ErrorAction SilentlyContinue) {
        Write-Host "Service $Name already exists -- stopping and removing before reinstalling."
        & $NssmExe stop $Name confirm | Out-Null
        & $NssmExe remove $Name confirm | Out-Null
    }

    & $NssmExe install $Name $VenvPython $ScriptFile
    & $NssmExe set $Name AppDirectory $ProjectDir
    & $NssmExe set $Name DisplayName $DisplayName
    & $NssmExe set $Name Description $Description
    & $NssmExe set $Name Start SERVICE_AUTO_START
    & $NssmExe set $Name AppStdout (Join-Path $LogDir "$Name.out.log")
    & $NssmExe set $Name AppStderr (Join-Path $LogDir "$Name.err.log")
    & $NssmExe set $Name AppRotateFiles 1
    & $NssmExe set $Name AppRotateBytes 5242880
    & $NssmExe set $Name AppRestartDelay 5000
    & $NssmExe start $Name
    Write-Host "Installed and started $Name."
}

Install-PtService -Name "PaperTiger-Watchdog" -ScriptFile "watchdog.py" `
    -DisplayName "PaperTiger Watchdog" `
    -Description "PaperTiger: watches runtime_state.json's heartbeat; its ONLY power is creating the HALT kill file if the engine hangs. Never trades."

Install-PtService -Name "PaperTiger-Dashboard" -ScriptFile "dashboard.py" `
    -DisplayName "PaperTiger Dashboard" `
    -Description "PaperTiger: status page at http://127.0.0.1:8787 with a kill-switch control. Cannot place a trade."

Install-PtService -Name "PaperTiger-Engine" -ScriptFile "run.py" `
    -DisplayName "PaperTiger Engine" `
    -Description "PaperTiger: the live (paper by default) trading loop. Refuses to trade real money unless ALPACA_PAPER=false AND I_UNDERSTAND_THIS_IS_REAL_MONEY=yes are both set in .env."

# -- Daily research task (Task Scheduler, not NSSM -- a one-shot job that
#    runs and exits, not a long-running process) --
$TaskName = "PaperTiger-DailyResearch"
$RetrainScript = Join-Path $PSScriptRoot "daily_retrain.ps1"

Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue

$Action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$RetrainScript`""
$Trigger = New-ScheduledTaskTrigger -Daily -At 6:00AM
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 5)
$Principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Limited

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Principal $Principal `
    -Description "PaperTiger: daily rerun of backtest.py + walkforward.py + train_ml_signal.py against fresh data." | Out-Null

Write-Host ""
Write-Host "Registered scheduled task $TaskName (daily at 6:00 AM)."

# -- Self-test task: runs the full unit test suite at system startup AND
#    on a recurring schedule, to catch environment drift in an unattended
#    deployment (see selftest.py's module docstring). Two triggers on one
#    task: AtStartup, plus a repeating interval thereafter. --
$SelfTestTaskName = "PaperTiger-SelfTest"
Unregister-ScheduledTask -TaskName $SelfTestTaskName -Confirm:$false -ErrorAction SilentlyContinue

$SelfTestAction = New-ScheduledTaskAction -Execute $VenvPython -Argument "selftest.py" -WorkingDirectory $ProjectDir
$StartupTrigger = New-ScheduledTaskTrigger -AtStartup
$RepeatTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date) `
    -RepetitionInterval (New-TimeSpan -Hours 4) -RepetitionDuration (New-TimeSpan -Days 3650)
$SelfTestSettings = New-ScheduledTaskSettingsSet -StartWhenAvailable
$SelfTestPrincipal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Limited

Register-ScheduledTask -TaskName $SelfTestTaskName -Action $SelfTestAction `
    -Trigger @($StartupTrigger, $RepeatTrigger) -Settings $SelfTestSettings -Principal $SelfTestPrincipal `
    -Description "PaperTiger: runs the full unit test suite at startup and every 4 hours; writes selftest_results.json for the dashboard's Status tab." | Out-Null

Write-Host "Registered scheduled task $SelfTestTaskName (at startup, then every 4 hours)."

# Run it once now too, synchronously, so the dashboard's Status tab has
# real data immediately instead of an empty state.
Write-Host ""
Write-Host "Running the self-test once now..."
& $VenvPython (Join-Path $ProjectDir "selftest.py")

# -- Tray notifier: shows a system-tray balloon for HALT/FLATTEN/watchdog
#    events while you're logged in (see notify.py + tray_notifier.ps1).
#    MUST run in the interactive session, not as an NSSM service -- Windows
#    services (Session 0) cannot show desktop UI. Registered as an "At Log
#    On" task instead, with NO -Principal override, so it runs as YOU
#    (whoever is running this install script), not SYSTEM. --
$TrayTaskName = "PaperTiger-TrayNotifier"
Unregister-ScheduledTask -TaskName $TrayTaskName -Confirm:$false -ErrorAction SilentlyContinue

$TrayScript = Join-Path $PSScriptRoot "tray_notifier.ps1"
$TrayAction = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$TrayScript`""
$TrayTrigger = New-ScheduledTaskTrigger -AtLogOn
$TraySettings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Days 3650)

Register-ScheduledTask -TaskName $TrayTaskName -Action $TrayAction -Trigger $TrayTrigger -Settings $TraySettings `
    -Description "PaperTiger: shows a system-tray balloon notification when something worth knowing about happens (kill switch, circuit breaker, watchdog) while you're logged in. Runs as you, not SYSTEM -- Windows services can't show desktop UI." | Out-Null

Write-Host "Registered scheduled task $TrayTaskName (at log on)."
Write-Host "Starting it now for your current session..."
Start-ScheduledTask -TaskName $TrayTaskName

Write-Host ""
Write-Host "All done. Check status with:"
Write-Host "  Get-Service PaperTiger-*"
Write-Host "  Get-ScheduledTask -TaskName PaperTiger-DailyResearch, PaperTiger-SelfTest, PaperTiger-TrayNotifier"
Write-Host ""
Write-Host "Stop everything at any time with the dashboard's kill switch, or"
Write-Host "'nssm stop PaperTiger-Engine'. To fully remove, run uninstall_services.ps1."
