#!/usr/bin/env bash
# Start DiffusionGemma: engine + system-tray icon (Electron app). Chat window opens from the tray.
# Needs a desktop session. For a pure HEADLESS engine (e.g. on a server / for Hermes) run instead:
#   PYTHONPATH="$ROOT" python3 -m agent.openai_server --host 0.0.0.0 --port 8787
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
npm --prefix "$ROOT/gui" start
