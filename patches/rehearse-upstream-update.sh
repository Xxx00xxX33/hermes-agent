#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

KEEP_TEMP=0
FETCH_REMOTE=1

usage() {
  cat <<'EOF'
Usage: patches/rehearse-upstream-update.sh [--keep-temp] [--no-fetch]

Verify the local customization bundle, then rehearse patch applicability
against a clean clone of the configured upstream baseline.

Options:
  --keep-temp  Keep the temporary clean clone for inspection.
  --no-fetch   Skip 'git fetch --all --prune' in the source repo.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep-temp)
      KEEP_TEMP=1
      shift
      ;;
    --no-fetch)
      FETCH_REMOTE=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
done

resolve_base_ref() {
  if [[ -n "${HERMES_LOCAL_CUSTOMIZATION_BASE_REF:-}" ]]; then
    printf '%s\n' "$HERMES_LOCAL_CUSTOMIZATION_BASE_REF"
  else
    python3 -m patches.local_customization_manifest --repo-root "$1" resolve-base-ref
  fi
}

resolve_rehearsal_remote_url() {
  if [[ -n "${HERMES_LOCAL_CUSTOMIZATION_UPSTREAM_URL:-}" ]]; then
    printf '%s\n' "$HERMES_LOCAL_CUSTOMIZATION_UPSTREAM_URL"
    return 0
  fi

  if [[ -f "$REPO_ROOT/metadata.json" ]]; then
    local metadata_url
    metadata_url="$(python3 - <<'PY' "$REPO_ROOT/metadata.json"
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
try:
    data = json.loads(path.read_text(encoding='utf-8'))
except Exception:
    print('', end='')
    raise SystemExit(0)
print(str(data.get('upstream_remote') or ''), end='')
PY
)"
    if [[ -n "$metadata_url" ]]; then
      printf '%s\n' "$metadata_url"
      return 0
    fi
  fi

  git remote get-url "$SOURCE_REMOTE"
}

patch_state_in_repo() {
  local repo_root="$1"
  local patch_file="$2"

  if git -C "$repo_root" apply --reverse --check "$patch_file" >/dev/null 2>&1; then
    printf 'already-applied'
  elif git -C "$repo_root" apply --check --3way "$patch_file" >/dev/null 2>&1; then
    printf 'applicable'
  else
    printf 'conflict'
  fi
}

SOURCE_BASE_REF="$(resolve_base_ref "$REPO_ROOT")"
if [[ "$SOURCE_BASE_REF" != */* ]]; then
  echo "Expected a remote-tracking base ref such as upstream/main or origin/main; got: $SOURCE_BASE_REF" >&2
  exit 2
fi
SOURCE_REMOTE="${SOURCE_BASE_REF%%/*}"
SOURCE_BRANCH="${SOURCE_BASE_REF#*/}"
REMOTE_URL="$(resolve_rehearsal_remote_url)"
OVERLAY_EXPORT_REPO=0
if [[ -f "$REPO_ROOT/metadata.json" ]]; then
  OVERLAY_EXPORT_REPO=1
fi

if (( FETCH_REMOTE == 1 )); then
  git fetch --all --prune
fi

TMPDIR="$(mktemp -d /tmp/hermes-upstream-rehearsal.XXXXXX)"
CLEAN_REPO="$TMPDIR/upstream"
cleanup() {
  if (( KEEP_TEMP == 0 )); then
    rm -rf "$TMPDIR"
  fi
}
trap cleanup EXIT

echo "Hermes local customization upstream rehearsal"
echo " - source repo: $REPO_ROOT"
echo " - source base ref: $SOURCE_BASE_REF"
echo " - rehearsal remote URL: $REMOTE_URL"
echo " - temp clone: $CLEAN_REPO"
echo

echo "Verifying local customization governance in source repo..."
if (( OVERLAY_EXPORT_REPO == 1 )); then
  echo "Detected sanitized overlay export; skipping source-repo strict/ahead gates."
else
  python3 -m patches.local_customization_manifest --repo-root "$REPO_ROOT" verify --strict
  python3 -m patches.local_customization_manifest --repo-root "$REPO_ROOT" verify-ahead --base-ref "$SOURCE_BASE_REF"
fi

echo
echo "Cloning clean upstream baseline..."
git clone --quiet --origin origin --branch "$SOURCE_BRANCH" --single-branch "$REMOTE_URL" "$CLEAN_REPO"
CLEAN_BASE_REF="$(python3 -m patches.local_customization_manifest --repo-root "$CLEAN_REPO" resolve-base-ref --prefer "origin/$SOURCE_BRANCH")"

echo "Resolved clean-clone base ref: $CLEAN_BASE_REF"
echo
echo "Patch applicability against clean upstream:"

already_applied_count=0
applicable_count=0
conflict_count=0
conflicts=()

while IFS='|' read -r patch_id patch_rel; do
  patch_file="$REPO_ROOT/$patch_rel"
  state="$(patch_state_in_repo "$CLEAN_REPO" "$patch_file")"
  printf ' - %-32s %s\n' "$patch_id" "$state"

  case "$state" in
    already-applied)
      already_applied_count=$((already_applied_count + 1))
      ;;
    applicable)
      applicable_count=$((applicable_count + 1))
      ;;
    conflict)
      conflict_count=$((conflict_count + 1))
      conflicts+=("$patch_id:$patch_rel")
      git -C "$CLEAN_REPO" apply --check --verbose "$patch_file" 2>&1 \
        | grep -E '^(Checking patch|error:)' \
        | sed 's/^/     /' || true
      ;;
  esac
done < <(python3 -m patches.local_customization_manifest --repo-root "$REPO_ROOT" bundle-lines)

echo
echo "Summary:"
echo " - already-applied: $already_applied_count"
echo " - applicable:      $applicable_count"
echo " - conflict:        $conflict_count"

if (( conflict_count > 0 )); then
  echo
  echo "Conflicting patch artifacts need refresh before an upstream update:" >&2
  printf ' - %s\n' "${conflicts[@]}" >&2
  if (( KEEP_TEMP == 1 )); then
    echo "Temporary clone preserved at: $CLEAN_REPO" >&2
  fi
  exit 1
fi

echo
if (( KEEP_TEMP == 1 )); then
  echo "Rehearsal completed successfully. Temporary clone preserved at: $CLEAN_REPO"
else
  echo "Rehearsal completed successfully."
fi
