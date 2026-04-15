#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BASE_REF="${HERMES_LOCAL_CUSTOMIZATION_BASE_REF:-upstream/main}"

MODE="apply"
STRICT_CHECK=0
RUN_TESTS=1
if [[ "${1:-}" == "--check" ]]; then
  MODE="check"
  STRICT_CHECK=1
  shift || true
elif [[ "${1:-}" == "--status" ]]; then
  MODE="check"
  STRICT_CHECK=0
  shift || true
fi
if [[ "${1:-}" == "--no-tests" ]]; then
  RUN_TESTS=0
  shift || true
fi
if [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--check|--status] [--no-tests]" >&2
  exit 2
fi

mapfile -t PATCHES < <(
  python3 -m patches.local_customization_manifest --repo-root "$REPO_ROOT" bundle-lines
)

LOCAL_HELPERS=(
  "patches/ensure-hermes-tmux-scrollkeys.sh"
)

TESTS=(
  tests/cli/test_cli_status_bar.py
  tests/hermes_cli/test_skin_engine.py
  tests/cli/test_cli_skin_integration.py
  tests/test_cli_skin_integration.py
  tests/hermes_cli/test_session_browse.py
  tests/cli/test_resume_display.py
  tests/cli/test_cli_init.py
  tests/agent/test_context_compressor.py
  tests/run_agent/test_compression_feasibility.py
  tests/run_agent/test_context_pressure.py
  tests/run_agent/test_long_context_tier_429.py
)

manifest_verify() {
  local strict_flag=()
  if (( STRICT_CHECK == 1 )) || [[ "$MODE" == "apply" ]]; then
    strict_flag+=(--strict)
  fi
  python3 -m patches.local_customization_manifest \
    --repo-root "$REPO_ROOT" \
    verify \
    "${strict_flag[@]}"
  python3 -m patches.local_customization_manifest \
    --repo-root "$REPO_ROOT" \
    verify-ahead \
    --base-ref "$BASE_REF"
}

patch_state() {
  local patch_file="$1"
  if git apply --reverse --check "$patch_file" >/dev/null 2>&1; then
    printf 'already-applied'
  elif git apply --check --3way "$patch_file" >/dev/null 2>&1; then
    printf 'applicable'
  else
    printf 'conflict'
  fi
}

activate_venv_if_present() {
  if [[ -f venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
  elif [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
  fi
}

failures=()
applied_any=0

echo "Hermes local customization patch bundle ($MODE mode)"
manifest_verify
for entry in "${PATCHES[@]}"; do
  name="${entry%%|*}"
  patch_file="${entry#*|}"
  state="$(patch_state "$patch_file")"
  printf ' - %-24s %s\n' "$name" "$state"

  if [[ "$MODE" == "apply" ]]; then
    case "$state" in
      already-applied)
        ;;
      applicable)
        echo "   applying: $patch_file"
        git apply --3way "$patch_file"
        applied_any=1
        ;;
      conflict)
        failures+=("$name:$patch_file")
        ;;
    esac
  elif [[ "$state" == "conflict" ]]; then
    failures+=("$name:$patch_file")
  fi
done

if (( ${#failures[@]} > 0 )); then
  echo
  echo "The following local customizations need manual review or patch refresh:" >&2
  printf ' - %s\n' "${failures[@]}" >&2
  if (( STRICT_CHECK == 1 )) || [[ "$MODE" == "apply" ]]; then
    exit 1
  fi
fi

if [[ "$MODE" == "check" ]]; then
  echo
  echo "Status check completed."
  exit 0
fi

if (( ${#LOCAL_HELPERS[@]} > 0 )); then
  echo
  echo "Applying local helper scripts..."
  for helper in "${LOCAL_HELPERS[@]}"; do
    echo " - $helper"
    "$helper"
  done
fi

if (( RUN_TESTS == 0 )); then
  echo
  echo "Patches processed. Tests skipped by request."
  exit 0
fi

activate_venv_if_present
if command -v pytest >/dev/null 2>&1; then
  echo
  echo "Running combined targeted verification tests..."
  pytest "${TESTS[@]}" -q
else
  echo
  echo "pytest not found; patch application finished without automated tests."
fi
