#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_FILE="$REPO_ROOT/patches/hermes-cli-working-indicator.patch"

cd "$REPO_ROOT"

echo "Checking patch applicability: $PATCH_FILE"
git apply --check --3way "$PATCH_FILE"
echo "Applying patch..."
git apply --3way "$PATCH_FILE"

echo "Patch applied successfully."
if [ -f venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
elif [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

if command -v pytest >/dev/null 2>&1; then
  echo "Running targeted verification tests..."
  pytest tests/cli/test_cli_status_bar.py tests/cli/test_cli_skin_integration.py tests/test_cli_skin_integration.py -q
else
  echo "pytest not found; skip tests."
fi
