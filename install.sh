#!/usr/bin/env bash
set -euo pipefail

# 420vision: One-Line Terminal Installer for macOS & Linux
# Installs to ~/.local/share/420vision and links ~/.local/bin/420vision

REPO="ganendraditya/20-20-20-20-vision"
INSTALL_DIR="${HOME}/.local/share/420vision"
BIN_DIR="${HOME}/.local/bin"
LAUNCH_AGENTS_DIR="${HOME}/Library/LaunchAgents"
PLIST_FILE="${LAUNCH_AGENTS_DIR}/com.ganendraditya.vision420.plist"

echo "🌸 Installing 420vision (20-20-20-20 Vision Assistant)..."

# Detect OS & Architecture
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"

if [ "$OS" != "darwin" ]; then
    echo "❌ Error: 420vision currently provides native desktop builds for macOS and Windows."
    exit 1
fi

if [ "$ARCH" != "arm64" ]; then
    echo "⚠️ Warning: Intel Mac (x86_64) detected. Compiling/running via Rosetta or Universal binary."
fi

# Fetch latest release tag
echo "🔍 Fetching latest release from GitHub..."
RELEASE_JSON=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" 2>/dev/null || true)

if [ -z "$RELEASE_JSON" ] || ! echo "$RELEASE_JSON" | grep -q "tag_name"; then
    echo "⚠️ No published release tag found. Using fallback master/v0.1.0."
    LATEST_TAG="v0.1.0"
else
    LATEST_TAG=$(echo "$RELEASE_JSON" | grep '"tag_name":' | head -n 1 | cut -d '"' -f 4)
fi

echo "📦 Selected version: ${LATEST_TAG}"

TAR_URL="https://github.com/${REPO}/releases/download/${LATEST_TAG}/420vision-macos-arm64.tar.gz"

# Create directories
mkdir -p "${INSTALL_DIR}"
mkdir -p "${BIN_DIR}"

# Download & Extract
echo "⬇️ Downloading native macOS binary..."
TEMP_TAR="/tmp/420vision-${LATEST_TAG}.tar.gz"

if curl -fL --progress-bar -o "${TEMP_TAR}" "${TAR_URL}" 2>/dev/null; then
    tar -xzf "${TEMP_TAR}" -C "${INSTALL_DIR}"
    rm -f "${TEMP_TAR}"
else
    echo "⚠️ Pre-built release binary not yet published on GitHub Releases."
    echo "🔨 Falling back to local/manual installation layout."
    
    # If ran from inside the repo clone, copy current binary if available
    if [ -f "src-tauri/target/release/420vision" ]; then
        cp "src-tauri/target/release/420vision" "${INSTALL_DIR}/420vision"
        mkdir -p "${INSTALL_DIR}/models"
        [ -f "models/facemesh.onnx" ] && cp "models/facemesh.onnx" "${INSTALL_DIR}/models/"
        [ -f "models/blazeface.onnx" ] && cp "models/blazeface.onnx" "${INSTALL_DIR}/models/"
    elif [ -f "dist-release/420vision" ]; then
        cp -r dist-release/* "${INSTALL_DIR}/"
    else
        echo "❌ Error: Please build with 'npm run tauri build -- --no-bundle' or download release tag."
        exit 1
    fi
fi

chmod +x "${INSTALL_DIR}/420vision"

# Symlink launcher
ln -sf "${INSTALL_DIR}/420vision" "${BIN_DIR}/420vision"

# Ensure ~/.local/bin is in PATH
SHELL_CONFIG=""
case "${SHELL:-}" in
    */zsh) SHELL_CONFIG="${HOME}/.zshrc" ;;
    */bash) SHELL_CONFIG="${HOME}/.bashrc" ;;
esac

if [ -n "$SHELL_CONFIG" ] && [ -f "$SHELL_CONFIG" ]; then
    if ! grep -Fq 'export PATH="$HOME/.local/bin:$PATH"' "$SHELL_CONFIG" 2>/dev/null; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_CONFIG"
    fi
fi

# Setup macOS LaunchAgent for auto-start at login
echo "🚀 Configuring auto-start in Menu Bar..."
mkdir -p "${LAUNCH_AGENTS_DIR}"
cat <<EOF > "${PLIST_FILE}"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.ganendraditya.vision420</string>
    <key>ProgramArguments</key>
    <array>
        <string>${INSTALL_DIR}/420vision</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <false/>
</dict>
</plist>
EOF

# Load and start LaunchAgent (RunAtLoad=true automatically starts the daemon in Menu Bar)
launchctl unload "${PLIST_FILE}" 2>/dev/null || true
launchctl load -w "${PLIST_FILE}" 2>/dev/null || true

echo ""
echo "=========================================================="
echo "✨ 420vision successfully installed!"
echo "📍 Installed to: ${INSTALL_DIR}"
echo "🔗 CLI Launcher: ${BIN_DIR}/420vision"
echo "🌸 The app is now running in your Menu Bar (top right)."
echo "=========================================================="
