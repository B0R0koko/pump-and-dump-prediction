#!/usr/bin/env bash
# Install the complete research dataset; orchestration lives in the Python package.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

if ! command -v poetry >/dev/null 2>&1; then
    printf 'Poetry is required. Install Poetry, then run poetry install in %s.\n' "$PROJECT_ROOT" >&2
    exit 1
fi

# --project selects the environment without changing how relative data paths resolve.
exec poetry --project "$PROJECT_ROOT" run python -m installer "$@"
