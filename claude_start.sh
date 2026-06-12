#!/usr/bin/env bash
set -euo pipefail

NODE_MAJOR="${NODE_MAJOR:-24}"

echo "==> Detecting system..."
if [ -f /etc/os-release ]; then
  . /etc/os-release
  echo "OS: ${PRETTY_NAME:-unknown}"
else
  echo "Cannot detect OS: /etc/os-release not found"
  exit 1
fi

if ! command -v apt-get >/dev/null 2>&1; then
  echo "This script is for Debian/Ubuntu systems with apt-get."
  echo "If your server is CentOS/RHEL, tell me the output of: cat /etc/os-release"
  exit 1
fi

echo "==> Installing prerequisites..."
apt-get update
apt-get install -y ca-certificates curl gnupg

echo "==> Installing Node.js ${NODE_MAJOR}.x from NodeSource..."
curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR}.x" | bash -

echo "==> Installing nodejs..."
apt-get install -y nodejs

echo "==> Versions:"
node -v
npm -v
npm install -g @anthropic-ai/claude-code

export ANTHROPIC_BASE_URL=https://api.llm.ustc.edu.cn
export ANTHROPIC_AUTH_TOKEN=sk-uKDVtYcmnDB5rw0NBnqKpw
export ANTHROPIC_MODEL=deepseek-v4-pro[1m]

claude --version
claude
