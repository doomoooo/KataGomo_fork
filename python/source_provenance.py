"""Resolve a locked source identity from a Git tree or release archive."""

from __future__ import annotations

import pathlib
import re
import subprocess


REVISION_MARKER = ".katago-source-revision"


def clean_source_revision(root: pathlib.Path) -> tuple[str, str]:
    """Return ``(revision, source)`` while rejecting dirty or ambiguous input."""

    root = root.resolve()
    git_revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        text=True,
        capture_output=True,
    )
    if git_revision.returncode == 0:
        revision = git_revision.stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--short"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        if status:
            raise RuntimeError(f"source tree must be clean: {root}")
        source = "git"
    else:
        marker = root / REVISION_MARKER
        if not marker.is_file():
            raise RuntimeError(
                f"source identity unavailable: {root} has neither Git metadata "
                f"nor {REVISION_MARKER}"
            )
        revision = marker.read_text(encoding="utf-8").strip()
        source = "archive-marker"
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError(f"invalid source revision for {root}: {revision!r}")
    marker = root / REVISION_MARKER
    if marker.is_file():
        recorded = marker.read_text(encoding="utf-8").strip()
        if recorded != revision:
            raise RuntimeError(
                f"source marker mismatch for {root}: {recorded} != {revision}"
            )
    return revision, source
