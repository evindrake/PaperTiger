<#
uninstall_services.ps1 -- removes everything install_services.ps1 set up:
the three PaperTiger-* Windows services and the PaperTiger-DailyResearch /
PaperTiger-SelfTest / PaperTiger-TrayNotifier scheduled tasks. Run from an
ELEVATED (Administrator) PowerShell prompt.

This only stops/removes the SERVICE REGISTRATIONS -- it does not touch any
project files, results, or your .env. Your Alpaca account and any open
paper positions are entirely unaffected; if you want to also flatten open
positions, use the dashboard's kill switch (FLATTEN) before or after
running this.
#>

#Requires -RunAsAdministrator

function Resolve-NssmPath {
    $cmd = Get-Command nssm.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $found = Get-ChildItem -Path "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Filter "nssm.exe" -Recurse -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -like "*win64*" } | Select-Object -First 1
    if ($found) { return $found.FullName }
    throw "nssm.exe not found -- if the services are already gone, that's fine, nothing to do."
}

$NssmExe = Resolve-NssmPath

foreach ($name in @("PaperTiger-Watchdog", "PaperTiger-Dashboard", "PaperTiger-Engine")) {
    if (Get-Service -Name $name -ErrorAction SilentlyContinue) {
        Write-Host "Stopping and removing $name..."
        & $NssmExe stop $name confirm | Out-Null
        & $NssmExe remove $name confirm | Out-Null
    } else {
        Write-Host "$name is not installed -- skipping."
    }
}

foreach ($taskName in @("PaperTiger-DailyResearch", "PaperTiger-SelfTest", "PaperTiger-TrayNotifier")) {
    if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
        try { Stop-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue } catch {}
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
        Write-Host "Removed scheduled task $taskName."
    } else {
        Write-Host "$taskName task not found -- skipping."
    }
}

# Stop-ScheduledTask doesn't reliably kill the tray notifier's PowerShell
# process (it runs detached in the interactive session) -- find and stop it
# directly by command line if it's still running.
Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like "*-File*tray_notifier.ps1*" } |
    ForEach-Object {
        Write-Host "Stopping lingering tray notifier process (PID $($_.ProcessId))..."
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }

Write-Host "Done."
