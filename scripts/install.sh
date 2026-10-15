#!/usr/bin/env bash
# Pull and run the data pipeline images; only Python and Docker are needed locally.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

if ! command -v python3 >/dev/null 2>&1; then
    printf 'Python 3.10 or newer is required to run the Docker installer.\n' >&2
    exit 1
fi

# Keep the caller's working directory so relative data/config paths retain their meaning.
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m installer "$@"
