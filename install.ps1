# 420vision: One-Line Terminal Installer for Windows (PowerShell)
# Installs to $env:LOCALAPPDATA\420vision and adds to user PATH

$ErrorActionPreference = "Stop"

$Repo = "ganendraditya/20-20-20-20-vision"
$InstallDir = Join-Path $env:LOCALAPPDATA "420vision"
$BinDir = Join-Path $InstallDir "bin"
$StartupDir = [Environment]::GetFolderPath("Startup")
$ShortcutPath = Join-Path $StartupDir "420vision.lnk"

Write-Host "🌸 Installing 420vision (20-20-20-20 Vision Assistant)..." -ForegroundColor Cyan

# Fetch latest release tag
Write-Host "🔍 Fetching latest release from GitHub..." -ForegroundColor Yellow
$LatestTag = "v0.1.0"
try {
    $Release = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases/latest" -Headers @{ "User-Agent" = "PowerShell" }
    if ($Release.tag_name) {
        $LatestTag = $Release.tag_name
    }
} catch {
    Write-Host "⚠️ No published release found. Using fallback $LatestTag." -ForegroundColor DarkYellow
}

Write-Host "📦 Selected version: $LatestTag" -ForegroundColor Green
$ZipUrl = "https://github.com/$Repo/releases/download/$LatestTag/420vision-windows-x64.zip"

# Create directories
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
New-Item -ItemType Directory -Force -Path $BinDir | Out-Null

$TempZip = Join-Path $env:TEMP "420vision-$LatestTag.zip"

Write-Host "⬇️ Downloading native Windows x64 binary..." -ForegroundColor Yellow
try {
    Invoke-WebRequest -Uri $ZipUrl -OutFile $TempZip
    Expand-Archive -Path $TempZip -DestinationPath $InstallDir -Force
    Remove-Item $TempZip -Force
} catch {
    Write-Host "⚠️ Pre-built release binary not yet published on GitHub Releases." -ForegroundColor DarkYellow
    Write-Host "🔨 Checking for local build artifacts..." -ForegroundColor DarkYellow

    $LocalExe = "src-tauri\target\x86_64-pc-windows-msvc\release\420vision.exe"
    if (Test-Path $LocalExe) {
        Copy-Item $LocalExe -Destination (Join-Path $InstallDir "420vision.exe") -Force
        New-Item -ItemType Directory -Force -Path (Join-Path $InstallDir "models") | Out-Null
        if (Test-Path "models\facemesh.onnx") {
            Copy-Item "models\facemesh.onnx" -Destination (Join-Path $InstallDir "models\facemesh.onnx") -Force
        }
    } else {
        Write-Error "❌ Error: Could not download release archive and no local binary found."
        exit 1
    }
}

# Create command wrapper in bin directory
$CmdWrapper = Join-Path $BinDir "420vision.cmd"
"@echo off`r`nstart `"`" `"$InstallDir\420vision.exe`" %*" | Out-File -FilePath $CmdWrapper -Encoding ASCII

# Add bin directory to User PATH if not present
$UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($UserPath -notlike "*$BinDir*") {
    [Environment]::SetEnvironmentVariable("Path", "$UserPath;$BinDir", "User")
    Write-Host "🔗 Added $BinDir to User PATH." -ForegroundColor Green
}

# Create Startup Shortcut
Write-Host "🚀 Configuring auto-start in Windows System Tray..." -ForegroundColor Yellow
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = Join-Path $InstallDir "420vision.exe"
$Shortcut.WorkingDirectory = $InstallDir
$Shortcut.Description = "420vision: 20-20-20-20 Eye Assistant"
$Shortcut.Save()

# Launch application immediately
Start-Process -FilePath (Join-Path $InstallDir "420vision.exe")

Write-Host ""
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "✨ 420vision successfully installed!" -ForegroundColor Green
Write-Host "📍 Installed to: $InstallDir"
Write-Host "🔗 CLI Launcher: 420vision"
Write-Host "🌸 The app is now running in your System Tray (bottom right)." -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
