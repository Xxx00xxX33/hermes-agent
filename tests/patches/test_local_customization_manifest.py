from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from patches.local_customization_manifest import (
    bundle_entries,
    find_uncovered_ahead_modifications,
    find_uncovered_tracked_modifications,
    load_manifest,
    validate_manifest,
)


def _init_git_repo(repo_root: Path) -> None:
    subprocess.run(["git", "init"], cwd=repo_root, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "tests@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Hermes Tests"], cwd=repo_root, check=True)


def _commit_all(repo_root: Path, message: str) -> None:
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", message], cwd=repo_root, check=True, capture_output=True, text=True)


def test_bundle_entries_follow_manifest_order(tmp_path: Path) -> None:
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    (patches_dir / "manifest.yaml").write_text(
        """
version: 1
bundle:
  - id: first
    patch: patches/first.patch
    purpose: first patch
  - id: second
    patch: patches/second.patch
    purpose: second patch
""".strip()
        + "\n",
        encoding="utf-8",
    )

    manifest = load_manifest(tmp_path)

    entries = bundle_entries(manifest)
    assert [entry["id"] for entry in entries] == ["first", "second"]
    assert [entry["patch"] for entry in entries] == [
        "patches/first.patch",
        "patches/second.patch",
    ]


def test_validate_manifest_reports_missing_patch_file(tmp_path: Path) -> None:
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    (patches_dir / "manifest.yaml").write_text(
        """
version: 1
bundle:
  - id: missing
    patch: patches/missing.patch
    purpose: missing patch file
""".strip()
        + "\n",
        encoding="utf-8",
    )

    errors = validate_manifest(tmp_path)

    assert any("patches/missing.patch" in error for error in errors)


def test_uncovered_tracked_modifications_ignore_patch_bundle_artifacts_and_report_real_gaps(
    tmp_path: Path,
) -> None:
    _init_git_repo(tmp_path)

    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    (tmp_path / "covered.py").write_text("print('covered')\n", encoding="utf-8")
    (tmp_path / "uncovered.py").write_text("print('uncovered')\n", encoding="utf-8")
    (patches_dir / "covered.patch").write_text(
        """
diff --git a/covered.py b/covered.py
--- a/covered.py
+++ b/covered.py
@@ -1 +1 @@
-print('covered')
+print('covered v2')
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (patches_dir / "manifest.yaml").write_text(
        """
version: 1
bundle:
  - id: covered
    patch: patches/covered.patch
    purpose: covers covered.py
""".strip()
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(["git", "add", "covered.py", "uncovered.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=tmp_path, check=True, capture_output=True, text=True)

    (tmp_path / "covered.py").write_text("print('covered local change')\n", encoding="utf-8")
    (tmp_path / "uncovered.py").write_text("print('uncovered local change')\n", encoding="utf-8")
    (patches_dir / "docs-only.patch").write_text(
        """
diff --git a/docs.md b/docs.md
--- a/docs.md
+++ b/docs.md
@@ -0,0 +1 @@
+notes
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (patches_dir / "manifest.yaml").write_text(
        """
version: 1
bundle:
  - id: covered
    patch: patches/covered.patch
    purpose: covers covered.py
  - id: docs-only
    patch: patches/docs-only.patch
    purpose: exercise patch artifacts being ignored
""".strip()
        + "\n",
        encoding="utf-8",
    )

    uncovered = find_uncovered_tracked_modifications(tmp_path)

    assert uncovered == ["uncovered.py"]


def test_patch_helper_code_is_not_treated_as_ignored_patch_artifact(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)

    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    (patches_dir / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")

    subprocess.run(["git", "add", "patches/helper.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=tmp_path, check=True, capture_output=True, text=True)

    (patches_dir / "helper.py").write_text("VALUE = 2\n", encoding="utf-8")

    uncovered = find_uncovered_tracked_modifications(tmp_path)

    assert uncovered == ["patches/helper.py"]


def test_unlisted_top_level_patch_does_not_count_toward_coverage(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)

    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    (tmp_path / "covered.py").write_text("print('covered')\n", encoding="utf-8")
    (tmp_path / "uncovered.py").write_text("print('uncovered')\n", encoding="utf-8")
    (patches_dir / "covered.patch").write_text(
        """
diff --git a/covered.py b/covered.py
--- a/covered.py
+++ b/covered.py
@@ -1 +1 @@
-print('covered')
+print('covered v2')
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (patches_dir / "stray.patch").write_text(
        """
diff --git a/uncovered.py b/uncovered.py
--- a/uncovered.py
+++ b/uncovered.py
@@ -1 +1 @@
-print('uncovered')
+print('stray patch should not count')
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (patches_dir / "manifest.yaml").write_text(
        """
version: 1
bundle:
  - id: covered
    patch: patches/covered.patch
    purpose: covers covered.py
""".strip()
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(["git", "add", "covered.py", "uncovered.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=tmp_path, check=True, capture_output=True, text=True)

    (tmp_path / "covered.py").write_text("print('covered local change')\n", encoding="utf-8")
    (tmp_path / "uncovered.py").write_text("print('uncovered local change')\n", encoding="utf-8")

    uncovered = find_uncovered_tracked_modifications(tmp_path)

    assert uncovered == ["uncovered.py"]


def test_find_uncovered_ahead_modifications_reports_local_only_gaps(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)

    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    (tmp_path / "covered.py").write_text("print('covered')\n", encoding="utf-8")
    (tmp_path / "uncovered.py").write_text("print('uncovered')\n", encoding="utf-8")
    (patches_dir / "covered.patch").write_text(
        """
diff --git a/covered.py b/covered.py
--- a/covered.py
+++ b/covered.py
@@ -1 +1 @@
-print('covered')
+print('covered v2')
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (patches_dir / "manifest.yaml").write_text(
        """
version: 1
bundle:
  - id: covered
    patch: patches/covered.patch
    purpose: covers covered.py
""".strip()
        + "\n",
        encoding="utf-8",
    )

    _commit_all(tmp_path, "initial")
    subprocess.run(["git", "branch", "-M", "main"], cwd=tmp_path, check=True)
    subprocess.run(["git", "checkout", "-b", "local/customizations"], cwd=tmp_path, check=True, capture_output=True, text=True)

    (tmp_path / "covered.py").write_text("print('covered local change')\n", encoding="utf-8")
    (tmp_path / "uncovered.py").write_text("print('uncovered local change')\n", encoding="utf-8")
    (patches_dir / "scratch.patch").write_text("scratch\n", encoding="utf-8")
    _commit_all(tmp_path, "local customizations")

    uncovered = find_uncovered_ahead_modifications(tmp_path, base_ref="main")

    assert uncovered == ["uncovered.py"]


def test_verify_ahead_cli_exits_nonzero_when_local_ahead_changes_are_uncovered(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)

    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    (tmp_path / "covered.py").write_text("print('covered')\n", encoding="utf-8")
    (tmp_path / "uncovered.py").write_text("print('uncovered')\n", encoding="utf-8")
    (patches_dir / "covered.patch").write_text(
        """
diff --git a/covered.py b/covered.py
--- a/covered.py
+++ b/covered.py
@@ -1 +1 @@
-print('covered')
+print('covered v2')
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (patches_dir / "manifest.yaml").write_text(
        """
version: 1
bundle:
  - id: covered
    patch: patches/covered.patch
    purpose: covers covered.py
""".strip()
        + "\n",
        encoding="utf-8",
    )

    _commit_all(tmp_path, "initial")
    subprocess.run(["git", "branch", "-M", "main"], cwd=tmp_path, check=True)
    subprocess.run(["git", "checkout", "-b", "local/customizations"], cwd=tmp_path, check=True, capture_output=True, text=True)

    (tmp_path / "covered.py").write_text("print('covered local change')\n", encoding="utf-8")
    (tmp_path / "uncovered.py").write_text("print('uncovered local change')\n", encoding="utf-8")
    _commit_all(tmp_path, "local customizations")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "patches.local_customization_manifest",
            "--repo-root",
            str(tmp_path),
            "verify-ahead",
            "--base-ref",
            "main",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "uncovered.py" in result.stderr
