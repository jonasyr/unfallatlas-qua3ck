import hashlib
import json
import subprocess
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import nbformat
from nbformat import NotebookNode

from unfallatlas.presentation.models import ExportMetadata, GitMetadata

BERLIN_TZ = ZoneInfo("Europe/Berlin")


def _canonical_json(notebook: NotebookNode) -> str:
    serialized = nbformat.writes(notebook, version=4)
    return json.dumps(
        json.loads(serialized),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _sha256(notebook: NotebookNode) -> str:
    return hashlib.sha256(_canonical_json(notebook).encode("utf-8")).hexdigest()


def snapshot_sha256(nb: NotebookNode) -> str:
    return _sha256(deepcopy(nb))


def source_sha256(nb: NotebookNode) -> str:
    notebook = deepcopy(nb)
    notebook.metadata.pop("widgets", None)
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        cell.outputs = []
        cell.execution_count = None
        for key in ("execution", "collapsed", "scrolled"):
            cell.metadata.pop(key, None)
    return _sha256(notebook)


def _git(args: list[str], root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            shell=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _repo_name(repo_root: Path) -> str:
    """Repository name for display - prefers the origin remote (works for
    any clone location/name), falls back to the working directory name if
    there is no remote configured (e.g. a local-only repo)."""
    remote_url = _git(["remote", "get-url", "origin"], repo_root)
    if remote_url:
        return Path(remote_url.rstrip("/")).stem
    return repo_root.resolve(strict=False).name


def read_git_metadata(repo_root: Path) -> GitMetadata:
    commit = _git(["rev-parse", "HEAD"], repo_root)
    short_commit = _git(["rev-parse", "--short=12", "HEAD"], repo_root)
    branch = _git(["branch", "--show-current"], repo_root)
    status = _git(["status", "--porcelain=v1"], repo_root)
    return GitMetadata(
        commit=commit or "unknown",
        short_commit=short_commit or "unknown",
        branch=branch or "unknown",
        repo_name=_repo_name(repo_root),
        dirty=bool(status),
    )


def build_export_metadata(repo_root: Path, now: datetime | None = None) -> ExportMetadata:
    exported_at = now or datetime.now(UTC)
    if exported_at.tzinfo is None or exported_at.utcoffset() is None:
        exported_at = exported_at.replace(tzinfo=UTC)
    exported_at = exported_at.astimezone(UTC).replace(microsecond=0)
    return ExportMetadata(
        exported_at=exported_at,
        exported_at_local=exported_at.astimezone(BERLIN_TZ).strftime("%d.%m.%Y, %H:%M:%S %Z"),
        git=read_git_metadata(repo_root),
    )
