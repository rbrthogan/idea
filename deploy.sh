#!/usr/bin/env bash
set -euo pipefail

# Keep invocation independent of the caller's current directory.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if ! command -v python3 >/dev/null 2>&1; then
    echo "Python 3 is required for the deployment helper." >&2
    exit 1
fi
exec python3 "$SCRIPT_DIR/scripts/deploy.py" "$@"
