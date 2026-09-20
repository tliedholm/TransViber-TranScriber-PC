# Builds the standalone release: PyInstaller onedir -> installer + portable zip.
# Run from the project root:
#   powershell -NoProfile -ExecutionPolicy Bypass -File packaging\build_release.ps1
# Add -SkipPyInstaller to rebuild only the installer/zip from an existing dist.

param([switch]$SkipPyInstaller)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$dist = Join-Path $root "packaging\dist\TransViber TransScriber"

if (-not $SkipPyInstaller) {
    & "$root\.venv\Scripts\python.exe" -m PyInstaller packaging\TransViber.spec `
        --noconfirm --workpath packaging\build --distpath packaging\dist
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }
}

if (-not (Test-Path "$dist\TransViber TransScriber.exe")) { throw "dist not found" }

# Ship the plain-language README in the app folder too.
Copy-Item "$root\packaging\README-FIRST.txt" $dist -Force

# Installer (Setup.exe + .bin slices — a single exe can't exceed ~4.2 GB)
$iscc = @("${env:LOCALAPPDATA}\Programs\Inno Setup 6\ISCC.exe",
          "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe") |
    Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $iscc) { throw "Inno Setup 6 (ISCC.exe) not found" }
if (Test-Path "$root\packaging\dist\installer") { Remove-Item -Recurse -Force "$root\packaging\dist\installer" }
& $iscc "$root\packaging\installer.iss"
if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed" }

# One shareable zip holding the installer parts. tar.exe (bsdtar) rather than
# Compress-Archive: PS 5.1's cmdlet mishandles multi-GB archives (no zip64).
$setupZip = Join-Path $root "packaging\dist\TransViber TransScriber Setup.zip"
if (Test-Path $setupZip) { Remove-Item $setupZip }
$parts = Get-ChildItem "$root\packaging\dist\installer" -Name
tar -C "$root\packaging\dist\installer" -a -cf $setupZip @($parts)
if ($LASTEXITCODE -ne 0) { throw "installer zip failed" }

# Portable zip
$zip = Join-Path $root "packaging\dist\TransViber TransScriber Portable.zip"
if (Test-Path $zip) { Remove-Item $zip }
tar -C "$root\packaging\dist" -a -cf $zip "TransViber TransScriber"
if ($LASTEXITCODE -ne 0) { throw "portable zip failed" }

Write-Host "`nDone:"
Get-ChildItem "$root\packaging\dist" | Where-Object { -not $_.PSIsContainer } |
    ForEach-Object { Write-Host ("  {0}  {1:N1} GB" -f $_.Name, ($_.Length / 1GB)) }
