<#
tray_notifier.ps1 -- shows a Windows system-tray balloon notification for
each new entry in notifications.json (written by notify.py whenever
engine.py or watchdog.py has something worth knowing about: a circuit
breaker trip, a kill-switch trigger, a stale-heartbeat watchdog
intervention).

Must run in YOUR interactive desktop session, NOT as an NSSM service --
Windows services run in Session 0 and are blocked from showing UI on the
desktop (a security boundary since Windows Vista). This is registered by
install_services.ps1 as a Scheduled Task triggered "At Log On" instead of
an NSSM service, so it starts automatically each time you log in and runs
under your own account (able to draw UI), not SYSTEM.

Uses only .NET's System.Windows.Forms.NotifyIcon (built into Windows via
PowerShell, no extra install, no external notification library) -- this is
what actually puts the icon in the system tray and pops the balloon.

This script never reads .env, never talks to Alpaca, and cannot trade or
touch the kill file -- its only capability is displaying a notification
someone else (notify.py) already decided was worth surfacing.
#>

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$ProjectDir = Split-Path -Parent $PSScriptRoot
$NotificationsFile = Join-Path $ProjectDir "notifications.json"

$icon = New-Object System.Windows.Forms.NotifyIcon
$icon.Icon = [System.Drawing.SystemIcons]::Information
$icon.Visible = $true
$icon.Text = "PaperTiger"

# Don't replay the entire backlog on startup -- only show notifications
# that arrive AFTER this script starts. Seed $lastSeenTs from whatever's
# already in the file (if anything).
$lastSeenTs = $null
if (Test-Path $NotificationsFile) {
    try {
        $existing = Get-Content $NotificationsFile -Raw | ConvertFrom-Json
        if ($existing -and $existing.Count -gt 0) {
            $lastSeenTs = $existing[-1].ts
        }
    } catch {
        # malformed/partially-written file -- ignore, treat as "nothing seen yet"
    }
}

while ($true) {
    if (Test-Path $NotificationsFile) {
        try {
            $notifications = Get-Content $NotificationsFile -Raw | ConvertFrom-Json
            if ($notifications) {
                foreach ($n in $notifications) {
                    if ($null -eq $lastSeenTs -or [string]$n.ts -gt [string]$lastSeenTs) {
                        $icon.BalloonTipTitle = "PaperTiger: " + $n.subject
                        $icon.BalloonTipText = $n.body
                        $icon.ShowBalloonTip(10000)
                        $lastSeenTs = $n.ts
                    }
                }
            }
        } catch {
            # transient read/parse error (e.g. caught mid-write) -- ignore, try again next loop
        }
    }
    Start-Sleep -Seconds 10
}
