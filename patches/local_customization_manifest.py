from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Iterable

import yaml

MANIFEST_VERSION = 1


def manifest_path(repo_root: Path) -> Path:
    return repo_root / "patches" / "manifest.yaml"


def load_manifest(repo_root: Path) -> dict:
    path = manifest_path(repo_root)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Manifest must parse to a mapping: {path}")
    return data


def bundle_entries(manifest: dict) -> list[dict]:
    entries = manifest.get("bundle") or []
    if not isinstance(entries, list):
        raise ValueError("Manifest key 'bundle' must be a list")
    normalized: list[dict] = []
    for idx, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"Manifest bundle entry #{idx} must be a mapping")
        normalized.append(entry)
    return normalized


def validate_manifest(repo_root: Path) -> list[str]:
    errors: list[str] = []
    try:
        manifest = load_manifest(repo_root)
    except Exception as exc:
        return [f"Failed to load patches/manifest.yaml: {exc}"]

    version = manifest.get("version")
    if version != MANIFEST_VERSION:
        errors.append(
            f"patches/manifest.yaml must declare version: {MANIFEST_VERSION} (got {version!r})"
        )

    seen_ids: set[str] = set()
    seen_patches: set[str] = set()
    for idx, entry in enumerate(bundle_entries(manifest), start=1):
        entry_id = str(entry.get("id") or "").strip()
        patch_rel = str(entry.get("patch") or "").strip()
        purpose = str(entry.get("purpose") or "").strip()

        if not entry_id:
            errors.append(f"Bundle entry #{idx} is missing a non-empty 'id'")
        elif entry_id in seen_ids:
            errors.append(f"Bundle entry id is duplicated: {entry_id}")
        else:
            seen_ids.add(entry_id)

        if not patch_rel:
            errors.append(f"Bundle entry #{idx} is missing a non-empty 'patch'")
        elif patch_rel in seen_patches:
            errors.append(f"Bundle entry patch is duplicated: {patch_rel}")
        else:
            seen_patches.add(patch_rel)
            patch_path = repo_root / patch_rel
            if not patch_path.exists():
                errors.append(f"Bundle entry references missing patch file: {patch_rel}")

        if not purpose:
            errors.append(f"Bundle entry #{idx} ({entry_id or 'unknown'}) is missing 'purpose'")

    return errors


def patch_inventory_paths(repo_root: Path) -> list[Path]:
    manifest = load_manifest(repo_root)
    paths: list[Path] = []
    for entry in bundle_entries(manifest):
        patch_rel = str(entry.get("patch") or "").strip()
        if not patch_rel:
            continue
        patch_path = repo_root / patch_rel
        if patch_path.exists():
            paths.append(patch_path)
    return paths


def patch_referenced_files(patch_path: Path) -> set[str]:
    referenced: set[str] = set()
    for line in patch_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("+++ b/"):
            rel = line[len("+++ b/") :].strip()
            if rel and rel != "/dev/null":
                referenced.add(rel)
    return referenced


def patch_coverage_index(repo_root: Path) -> dict[str, set[str]]:
    coverage: dict[str, set[str]] = {}
    for patch_path in patch_inventory_paths(repo_root):
        coverage[patch_path.name] = patch_referenced_files(patch_path)
    return coverage


def covered_files(repo_root: Path) -> set[str]:
    covered: set[str] = set()
    for files in patch_coverage_index(repo_root).values():
        covered.update(files)
    return covered


def _git_lines(repo_root: Path, *args: str) -> list[str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {stderr}")
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def tracked_modified_files(repo_root: Path) -> list[str]:
    unstaged = _git_lines(repo_root, "diff", "--name-only", "--")
    staged = _git_lines(repo_root, "diff", "--name-only", "--cached", "--")
    return sorted(set(unstaged) | set(staged))


def ahead_modified_files(repo_root: Path, base_ref: str = "upstream/main") -> list[str]:
    return _git_lines(repo_root, "diff", "--name-only", f"{base_ref}...HEAD", "--")


def is_patch_artifact(rel_path: str) -> bool:
    path = PurePosixPath(rel_path)
    if not path.parts or path.parts[0] != "patches":
        return False
    if "archive" in path.parts:
        return True
    if path.name == "manifest.yaml":
        return True
    return path.suffix == ".patch"


def find_uncovered_tracked_modifications(repo_root: Path) -> list[str]:
    coverage = covered_files(repo_root)
    uncovered = []
    for rel_path in tracked_modified_files(repo_root):
        if is_patch_artifact(rel_path):
            continue
        if rel_path not in coverage:
            uncovered.append(rel_path)
    return sorted(uncovered)


def find_uncovered_ahead_modifications(repo_root: Path, base_ref: str = "upstream/main") -> list[str]:
    coverage = covered_files(repo_root)
    uncovered = []
    for rel_path in ahead_modified_files(repo_root, base_ref=base_ref):
        if is_patch_artifact(rel_path):
            continue
        if rel_path not in coverage:
            uncovered.append(rel_path)
    return sorted(uncovered)


def bundle_lines(repo_root: Path) -> list[str]:
    manifest = load_manifest(repo_root)
    lines = []
    for entry in bundle_entries(manifest):
        lines.append(f"{entry['id']}|{entry['patch']}")
    return lines


def verify_repo(repo_root: Path) -> tuple[list[str], list[str]]:
    errors = validate_manifest(repo_root)
    uncovered = find_uncovered_tracked_modifications(repo_root)
    warnings: list[str] = []
    if uncovered:
        joined = "\n - ".join(uncovered)
        warnings.append(
            "Tracked modifications missing from every manifest-listed patch:\n - " + joined
        )
    return errors, warnings


def verify_ahead_repo(repo_root: Path, base_ref: str = "upstream/main") -> tuple[list[str], list[str]]:
    errors = validate_manifest(repo_root)
    warnings: list[str] = []
    if errors:
        return errors, warnings

    try:
        uncovered = find_uncovered_ahead_modifications(repo_root, base_ref=base_ref)
    except RuntimeError as exc:
        return [str(exc)], warnings

    if uncovered:
        joined = "\n - ".join(uncovered)
        warnings.append(
            f"Ahead-of-{base_ref} modifications missing from every manifest-listed patch:\n - "
            + joined
        )
    return errors, warnings


def _print_lines(lines: Iterable[str], *, stream=sys.stdout) -> None:
    for line in lines:
        print(line, file=stream)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local Hermes customization manifest helpers")
    parser.add_argument(
        "--repo-root",
        default=Path(__file__).resolve().parents[1],
        type=Path,
        help="Hermes repo root (default: parent of patches/)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("bundle-lines", help="Print bundle entries as id|path lines")
    verify_parser = subparsers.add_parser("verify", help="Validate manifest and coverage")
    verify_parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if uncovered tracked modifications are found",
    )
    verify_ahead_parser = subparsers.add_parser(
        "verify-ahead",
        help="Validate that ahead-of-base local customizations are covered by the manifest",
    )
    verify_ahead_parser.add_argument(
        "--base-ref",
        default="upstream/main",
        help="Git base ref to compare against (default: upstream/main)",
    )

    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve()

    if args.command == "bundle-lines":
        _print_lines(bundle_lines(repo_root))
        return 0

    if args.command == "verify":
        errors, warnings = verify_repo(repo_root)
        if errors:
            _print_lines(errors, stream=sys.stderr)
        if warnings:
            _print_lines(warnings, stream=sys.stderr)
        if errors:
            return 1
        if warnings and args.strict:
            return 1
        return 0

    if args.command == "verify-ahead":
        errors, warnings = verify_ahead_repo(repo_root, base_ref=args.base_ref)
        if errors:
            _print_lines(errors, stream=sys.stderr)
        if warnings:
            _print_lines(warnings, stream=sys.stderr)
        if errors or warnings:
            return 1
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
