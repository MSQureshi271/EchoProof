#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${1:-.venv}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "ERROR: Python 3.11 or 3.13 is required; '$PYTHON' was not found." >&2; exit 1
fi
PY_MINOR="$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$PY_MINOR" != "3.11" && "$PY_MINOR" != "3.13" ]]; then echo "ERROR: expected Python 3.11 or 3.13; found $PY_MINOR." >&2; exit 1; fi
if [[ "$VENV" != /* ]]; then VENV="$ROOT/$VENV"; fi
if [[ ! -x "$VENV/bin/python" ]]; then "$PYTHON" -m venv "$VENV"; fi
if [[ "$SKIP_INSTALL" != "1" ]]; then
  "$VENV/bin/python" -m pip install -r "$ROOT/requirements/app.lock.txt" || { echo "ERROR: dependency installation failed." >&2; exit 1; }
  "$VENV/bin/python" -m pip install --no-deps --no-build-isolation -e "$ROOT" || { echo "ERROR: editable EchoProof install failed." >&2; exit 1; }
fi
"$VENV/bin/python" "$ROOT/scripts/check_app_env.py" || { echo "ERROR: import or pinned-version check failed." >&2; exit 1; }
echo "Environment ready: $VENV. Activate with: source '$VENV/bin/activate'"
