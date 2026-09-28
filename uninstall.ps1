# 420vision: Clean Uninstaller for Windows (PowerShell)
param(
    [switch]$RemoveAllData
)

$InstallDir = Join-Path $env:LOCALAPPDATA "420vision"
$StartupDir = [Environment]::GetFolderPath("Startup")
$ShortcutPath = Join-Path $StartupDir "420vision.lnk"
$AppDataDir = Join-Path $env:APPDATA "420vision"

Write-Host "🗑️ Uninstalling 420vision..." -ForegroundColor Yellow

# 1. Kill running processes
Stop-Process -Name "420vision" -Force -ErrorAction SilentlyContinue

# 2. Remove Startup shortcut
if (Test-Path $ShortcutPath) {
    Remove-Item $ShortcutPath -Force
    Write-Host "✓ Removed Startup shortcut" -ForegroundColor Green
}

# 3. Remove bin from PATH
$BinDir = Join-Path $InstallDir "bin"
$UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($UserPath -like "*$BinDir*") {
    $NewPath = ($UserPath -split ";" | Where-Object { $_ -ne $BinDir }) -join ";"
    [Environment]::SetEnvironmentVariable("Path", $NewPath, "User")
    Write-Host "✓ Removed 420vision from User PATH" -ForegroundColor Green
}

# 4. Remove installation files
if (Test-Path $InstallDir) {
    Remove-Item $InstallDir -Recurse -Force
    Write-Host "✓ Removed application binaries" -ForegroundColor Green
}

# Optional purge
if ($RemoveAllData) {
    if (Test-Path $AppDataDir) {
        Remove-Item $AppDataDir -Recurse -Force
        Write-Host "✓ Purged local analytics and configuration" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "✨ 420vision has been cleanly uninstalled from your Windows system." -ForegroundColor Cyan
