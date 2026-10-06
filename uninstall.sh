#!/usr/bin/env bash
set -euo pipefail

# 420vision: Clean Uninstaller for macOS & Linux

INSTALL_DIR="${HOME}/.local/share/420vision"
BIN_DIR="${HOME}/.local/bin"
PLIST_FILE="${HOME}/Library/LaunchAgents/com.ganendraditya.vision420.plist"
CONFIG_DIR="${HOME}/Library/Application Support/420vision"
FALLBACK_CONFIG="${HOME}/.config/420vision"

echo "Uninstalling 420vision..."

# 1. Kill running processes
pkill -f "420vision" 2>/dev/null || true

# 2. Remove LaunchAgent
if [ -f "${PLIST_FILE}" ]; then
    launchctl unload "${PLIST_FILE}" 2>/dev/null || true
    rm -f "${PLIST_FILE}"
    echo "[OK] Removed LaunchAgent"
fi

# 3. Remove binary symlink
rm -f "${BIN_DIR}/420vision"
echo "[OK] Removed CLI launcher symlink"

# 4. Remove installation files
rm -rf "${INSTALL_DIR}"
echo "[OK] Removed application binaries"

# Optional purge
if [ "${1:-}" = "--purge" ]; then
    rm -rf "${CONFIG_DIR}"
    rm -rf "${FALLBACK_CONFIG}"
    echo "[OK] Purged local analytics and configuration"
fi

echo ""
echo "420vision has been cleanly uninstalled from your system."
