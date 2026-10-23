#!/usr/bin/env bash
# Run the complete XRI pipeline using config.json unless --config is supplied.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON="$PYTHON_BIN"
elif [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
    PYTHON="$SCRIPT_DIR/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON="python"
else
    printf '%s\n' 'XRI ERROR: Python not found. Install Python 3.10+ and requirements.txt, or set PYTHON_BIN to the Python executable.' >&2
    exit 127
fi

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    printf 'XRI ERROR: Python executable not found: %s\n' "$PYTHON" >&2
    exit 127
fi

exec "$PYTHON" -u "$SCRIPT_DIR/run_xri.py" --config "$SCRIPT_DIR/config.json" "$@"
