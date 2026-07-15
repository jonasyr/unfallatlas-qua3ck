from __future__ import annotations

import argparse
import logging
import sys
import webbrowser
from collections.abc import Sequence
from pathlib import Path

from unfallatlas.presentation.discovery import discover_notebooks, resolve_explicit_notebooks
from unfallatlas.presentation.manifest import (
    ManifestError,
    check_freshness,
    load_manifest,
    update_manifest,
    write_manifest_and_index,
)
from unfallatlas.presentation.metadata import build_export_metadata
from unfallatlas.presentation.models import (
    BatchResult,
    ExportResult,
    NotebookStatus,
    Severity,
)
from unfallatlas.presentation.rendering import render_notebook
from unfallatlas.presentation.validation import read_and_validate_notebook

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export saved Jupyter notebook outputs as offline HTML presentations."
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("notebooks", nargs="*", metavar="NOTEBOOK")
    selection.add_argument("--all", action="store_true", help="export every discovered notebook")
    selection.add_argument(
        "--check", action="store_true", help="check manifest freshness without rendering"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="publication directory (default: reports/presentation)",
    )
    parser.add_argument("--strict", action="store_true", help="fail notebooks with blockers")
    parser.add_argument(
        "--include-placeholders",
        action="store_true",
        help="render notebooks classified as placeholders",
    )
    parser.add_argument("--open", action="store_true", help="open successful output in a browser")
    return parser


def _repository_candidate(start: Path) -> Path | None:
    resolved = start.resolve(strict=False)
    if resolved.is_file():
        resolved = resolved.parent
    for candidate in (resolved, *resolved.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / ".git").exists():
            return candidate
    return None


def find_repo_root(start: Path | None = None) -> Path:
    root = _repository_candidate(Path.cwd() if start is None else Path(start))
    if root is None:
        root = _repository_candidate(Path(__file__))
    if root is None:
        raise RuntimeError("repository root not found (expected pyproject.toml and .git)")
    return root


def _output_relative_path(source: Path, notebooks_dir: Path) -> Path:
    return (
        source.resolve(strict=False)
        .relative_to(notebooks_dir.resolve(strict=False))
        .with_suffix(".html")
    )


def _failed_result(
    source: Path,
    output_root: Path,
    output_relative_path: Path,
    reason: str,
) -> ExportResult:
    return ExportResult(
        source=source,
        destination=output_root / "notebooks" / output_relative_path,
        status=NotebookStatus.INVALID,
        findings=(),
        size_bytes=0,
        error=reason,
    )


def _selected_notebooks(args: argparse.Namespace, repo_root: Path) -> tuple[Path, ...]:
    notebooks_dir = repo_root / "notebooks"
    if args.all:
        return discover_notebooks(notebooks_dir)
    return resolve_explicit_notebooks(tuple(Path(item) for item in args.notebooks), notebooks_dir)


def run_export(args: argparse.Namespace, repo_root: Path) -> BatchResult:
    repo_root = Path(repo_root).resolve(strict=False)
    notebooks_dir = repo_root / "notebooks"
    output_root = (
        Path(args.output_dir).resolve(strict=False)
        if args.output_dir is not None
        else repo_root / "reports" / "presentation"
    )
    selected = _selected_notebooks(args, repo_root)
    manifest_path = output_root / "manifest.json"
    manifest = load_manifest(manifest_path)
    results: list[ExportResult] = []
    successful_renders = 0

    for source in selected:
        output_relative_path = _output_relative_path(source, notebooks_dir)
        try:
            analysis = read_and_validate_notebook(source, repo_root)
        except Exception as error:
            LOGGER.exception("Could not validate notebook %s", source)
            results.append(
                _failed_result(
                    source,
                    output_root,
                    output_relative_path,
                    str(error) or type(error).__name__,
                )
            )
            continue

        destination = output_root / "notebooks" / output_relative_path
        if analysis.status is NotebookStatus.PLACEHOLDER and not args.include_placeholders:
            results.append(
                ExportResult(
                    source,
                    destination,
                    analysis.status,
                    analysis.findings,
                    0,
                    None,
                )
            )
            continue
        if args.strict and analysis.strict_blocked:
            results.append(
                ExportResult(
                    source,
                    destination,
                    analysis.status,
                    analysis.findings,
                    0,
                    "strict validation blocked export",
                )
            )
            continue

        try:
            metadata = build_export_metadata(repo_root)
            result = render_notebook(
                analysis,
                metadata,
                output_root,
                repo_root=repo_root,
                output_relative_path=output_relative_path,
            )
            if result.error is None:
                manifest = update_manifest(
                    manifest,
                    analysis,
                    result,
                    metadata,
                    repo_root,
                    output_root=output_root,
                )
        except Exception as error:
            LOGGER.exception("Could not export notebook %s", source)
            results.append(
                ExportResult(
                    source,
                    destination,
                    analysis.status,
                    analysis.findings,
                    0,
                    str(error) or type(error).__name__,
                )
            )
            continue
        results.append(result)
        if result.error is None:
            successful_renders += 1

    if successful_renders:
        try:
            write_manifest_and_index(
                manifest,
                output_root,
                notebooks_dir=notebooks_dir,
                repo_root=repo_root,
            )
        except Exception as error:
            LOGGER.exception("Could not publish presentation manifest and index")
            reason = str(error) or type(error).__name__
            for index in range(len(results)):
                if results[index].error is None and results[index].size_bytes:
                    previous = results[index]
                    results[index] = ExportResult(
                        previous.source,
                        previous.destination,
                        previous.status,
                        previous.findings,
                        previous.size_bytes,
                        f"publication failed: {reason}",
                        previous.assets,
                    )

    return BatchResult(tuple(results))


def _human_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    for suffix in ("KiB", "MiB", "GiB"):
        size_float = size / 1024
        if size_float < 1024 or suffix == "GiB":
            return f"{size_float:.1f} {suffix}"
        size = int(size_float)
    raise AssertionError("unreachable")


def _finding_summary(result: ExportResult) -> str:
    counts = {
        severity: sum(finding.severity is severity for finding in result.findings)
        for severity in Severity
    }
    return (
        f"{counts[Severity.INFO]} info, {counts[Severity.WARNING]} warnings, "
        f"{counts[Severity.ERROR]} errors"
    )


def _write_export_summary(batch: BatchResult) -> None:
    for result in batch.results:
        if result.error:
            outcome = f"failed — {result.error}"
        elif result.status is NotebookStatus.PLACEHOLDER and result.size_bytes == 0:
            outcome = "skipped placeholder"
        else:
            outcome = "exported"
        sys.stdout.write(
            f"{result.source.name} — {outcome} — {result.status.value} — "
            f"{_human_size(result.size_bytes)} — {_finding_summary(result)}\n"
        )


def _run_check(args: argparse.Namespace, repo_root: Path) -> int:
    output_root = (
        Path(args.output_dir).resolve(strict=False)
        if args.output_dir is not None
        else repo_root / "reports" / "presentation"
    )
    manifest_path = output_root / "manifest.json"
    if not manifest_path.is_file():
        sys.stderr.write(f"error: presentation manifest not found: {manifest_path}\n")
        return 1
    manifest = load_manifest(manifest_path)
    if not manifest.entries:
        sys.stderr.write(f"error: presentation manifest has no notebook entries: {manifest_path}\n")
        return 1
    freshness = check_freshness(
        manifest,
        notebooks_dir=repo_root / "notebooks",
        output_root=output_root,
        repo_root=repo_root,
    )
    for result in freshness:
        sys.stdout.write(f"{Path(result.source).name} — {result.state} — {result.status}\n")
    return 0 if all(result.state == "fresh" for result in freshness) else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        repo_root = find_repo_root()
        if args.check:
            return _run_check(args, repo_root)
        batch = run_export(args, repo_root)
    except ManifestError as error:
        sys.stderr.write(f"error: {error}\n")
        return 1
    except (OSError, RuntimeError, ValueError) as error:
        sys.stderr.write(f"error: {error}\n")
        return 2

    _write_export_summary(batch)
    if batch.successful and args.open:
        rendered = [result for result in batch.results if result.size_bytes > 0]
        if not rendered:
            return 0
        target = (
            rendered[0].destination
            if len(rendered) == 1
            else (
                Path(args.output_dir).resolve(strict=False)
                if args.output_dir is not None
                else repo_root / "reports" / "presentation"
            )
            / "index.html"
        )
        webbrowser.open(target.resolve(strict=False).as_uri())
    return 0 if batch.successful else 1
