<#
generate_tray_icon.ps1 -- draws PaperTiger's tray icon (scripts/papertiger.ico)
and writes it to disk. Run this once (or again, any time you want to tweak
the design) -- tray_notifier.ps1 just loads the resulting .ico file, it
never draws anything itself.

Same mark as dashboard.py's inline SVG banner (see _PAPER_TIGER_MARK there):
a rounded orange badge with three clipped diagonal black stripes -- a
literal, unambiguous "tiger stripes" mark chosen specifically because it
stays legible at a tiny 32x32 tray-icon size, which a more detailed/
naturalistic face would risk not doing.

Uses only .NET's System.Drawing (built into Windows via PowerShell, same
as tray_notifier.ps1 itself) -- no image library, no external asset tool.
#>

Add-Type -AssemblyName System.Drawing

$OutPath = Join-Path $PSScriptRoot "papertiger.ico"

$size = 64
$bitmap = New-Object System.Drawing.Bitmap($size, $size)
$g = [System.Drawing.Graphics]::FromImage($bitmap)
$g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$g.Clear([System.Drawing.Color]::Transparent)

$orange = [System.Drawing.Color]::FromArgb(255, 0xF9, 0x73, 0x16)
$dark = [System.Drawing.Color]::FromArgb(255, 0x1F, 0x22, 0x29)

# Rounded-rect badge outline, built from arcs + lines (System.Drawing has no
# built-in rounded-rect primitive) -- used both as the fill shape and as a
# clip region for the stripes so they never spill past the rounded corners.
function New-RoundedRectPath([float]$x, [float]$y, [float]$w, [float]$h, [float]$r) {
    $path = New-Object System.Drawing.Drawing2D.GraphicsPath
    $path.AddArc($x, $y, $r * 2, $r * 2, 180, 90)
    $path.AddArc($x + $w - $r * 2, $y, $r * 2, $r * 2, 270, 90)
    $path.AddArc($x + $w - $r * 2, $y + $h - $r * 2, $r * 2, $r * 2, 0, 90)
    $path.AddArc($x, $y + $h - $r * 2, $r * 2, $r * 2, 90, 90)
    $path.CloseFigure()
    return $path
}

$badgePath = New-RoundedRectPath 4 4 56 56 14

$g.SetClip($badgePath)
$orangeBrush = New-Object System.Drawing.SolidBrush($orange)
$g.FillRectangle($orangeBrush, 0, 0, $size, $size)

$darkBrush = New-Object System.Drawing.SolidBrush($dark)
# Three diagonal stripes, same coordinates as the dashboard's inline SVG
# mark -- clipped to the badge's rounded-rect region set above.
$stripes = @(
    @([System.Drawing.PointF]::new(15, -4), [System.Drawing.PointF]::new(21, -4), [System.Drawing.PointF]::new(7, 68), [System.Drawing.PointF]::new(1, 68)),
    @([System.Drawing.PointF]::new(29, -4), [System.Drawing.PointF]::new(35, -4), [System.Drawing.PointF]::new(21, 68), [System.Drawing.PointF]::new(15, 68)),
    @([System.Drawing.PointF]::new(43, -4), [System.Drawing.PointF]::new(49, -4), [System.Drawing.PointF]::new(35, 68), [System.Drawing.PointF]::new(29, 68))
)
foreach ($stripe in $stripes) {
    $g.FillPolygon($darkBrush, $stripe)
}
$g.ResetClip()

$pen = New-Object System.Drawing.Pen($dark, 2)
$g.DrawPath($pen, $badgePath)

$hicon = $bitmap.GetHicon()
$icon = [System.Drawing.Icon]::FromHandle($hicon)
$fileStream = [System.IO.File]::Create($OutPath)
$icon.Save($fileStream)
$fileStream.Close()

# GetHicon() allocates a native GDI icon handle that .NET's Icon wrapper
# does not own/free automatically -- DestroyIcon it explicitly to avoid
# leaking a handle (harmless for a one-shot script, but this is the
# documented-correct cleanup and costs nothing to do).
Add-Type -MemberDefinition '[DllImport("user32.dll")] public static extern bool DestroyIcon(IntPtr hIcon);' -Name "IconUtil" -Namespace "PaperTiger"
[PaperTiger.IconUtil]::DestroyIcon($hicon) | Out-Null

$g.Dispose()
$bitmap.Dispose()

Write-Host "Wrote $OutPath"
