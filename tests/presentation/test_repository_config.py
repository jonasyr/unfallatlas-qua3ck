import re
import shlex
import subprocess
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(relative_path: str) -> str:
    path = REPO_ROOT / relative_path
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )


def test_presentation_large_assets_are_narrowly_lfs_scoped() -> None:
    attrs = _read(".gitattributes")
    presentation_rules = [
        line
        for line in attrs.splitlines()
        if line.strip() and not line.lstrip().startswith("#") and "reports/presentation" in line
    ]

    assert presentation_rules == [
        "reports/presentation/assets/notebooks/** filter=lfs diff=lfs merge=lfs -text",
        "reports/presentation/assets/vendor/** filter=lfs diff=lfs merge=lfs -text",
    ]


def test_only_interrupted_presentation_writes_are_ignored() -> None:
    ignore = _read(".gitignore")
    presentation_rules = [
        line
        for line in ignore.splitlines()
        if line.strip() and not line.lstrip().startswith("#") and "reports/presentation" in line
    ]

    assert presentation_rules == ["reports/presentation/**/.tmp-*"]


def test_pages_workflow_has_relevant_triggers_and_concurrency() -> None:
    workflow = _read(".github/workflows/pages.yml")

    assert "branches: [main]" in workflow
    assert '      - "reports/presentation/**"' in workflow
    assert '      - ".github/workflows/pages.yml"' in workflow
    assert "workflow_dispatch:" in workflow
    assert "concurrency:\n  group: pages\n  cancel-in-progress: false" in workflow


def test_pages_workflow_uses_minimum_permissions() -> None:
    workflow = _read(".github/workflows/pages.yml")

    assert "permissions:\n  contents: read\n\nconcurrency:" in workflow
    assert (
        "    permissions:\n"
        "      contents: read\n"
        "      pages: write\n"
        "      id-token: write\n"
        "    runs-on: ubuntu-latest"
    ) in workflow


def test_pages_workflow_deploys_committed_lfs_artifact() -> None:
    workflow = _read(".github/workflows/pages.yml")

    assert "actions/checkout@v6" in workflow
    assert "          lfs: true" in workflow
    assert "actions/configure-pages@v6" in workflow
    assert "actions/upload-pages-artifact@v5" in workflow
    assert "          path: reports/presentation" in workflow
    assert "actions/deploy-pages@v5" in workflow


def test_pages_workflow_validates_complete_artifact_below_one_gibibyte() -> None:
    workflow = _read(".github/workflows/pages.yml")

    assert "test -f reports/presentation/index.html" in workflow
    assert "test -f reports/presentation/manifest.json" in workflow
    assert 'test "$(du -sm reports/presentation | cut -f1)" -lt 1024' in workflow


def test_pages_workflow_never_executes_or_tests_notebooks() -> None:
    workflow = _read(".github/workflows/pages.yml").lower()

    assert "jupyter" not in workflow
    assert "pytest" not in workflow
    assert "python" not in workflow
    assert "export_notebooks" not in workflow


def test_ci_installs_presentation_dependencies() -> None:
    workflow = _read(".github/workflows/ci.yml")

    assert "run: uv sync --extra dev --extra geo --extra presentation" in workflow


def test_large_file_hook_remains_at_five_mibibytes() -> None:
    config = _read(".pre-commit-config.yaml")

    assert "args: [--maxkb=5120]" in config


def test_nbstripout_preserves_saved_output_fixtures_but_covers_source_notebooks() -> None:
    config = _read(".pre-commit-config.yaml")
    hook = re.search(r"(?ms)^      - id: nbstripout\n(?P<body>(?:^        .*\n)+)", config)

    assert hook is not None
    body = hook.group("body")
    files_match = re.search(r"^        files: (.+)$", body, flags=re.MULTILINE)
    exclude_match = re.search(r"^        exclude: (.+)$", body, flags=re.MULTILINE)
    assert files_match is not None
    assert exclude_match is not None
    assert exclude_match.group(1) == "^tests/presentation/fixtures/"

    files_pattern = re.compile(files_match.group(1))
    exclude_pattern = re.compile(exclude_match.group(1))
    source_notebook = "notebooks/02_U_Phase.ipynb"
    output_fixture = "tests/presentation/fixtures/gallery.ipynb"

    assert files_pattern.search(source_notebook)
    assert not exclude_pattern.search(source_notebook)
    assert files_pattern.search(output_fixture)
    assert exclude_pattern.search(output_fixture)


def test_browser_checks_are_opt_in() -> None:
    pyproject = tomllib.loads(_read("pyproject.toml"))
    addopts = shlex.split(pyproject["tool"]["pytest"]["ini_options"]["addopts"])
    marker_index = addopts.index("-m")

    assert addopts[marker_index + 1] == "not browser"
    assert "--cov=src/unfallatlas" in addopts
    assert "--cov-report=xml" in addopts
    assert "--cov-report=term-missing" in addopts


def test_committed_presentation_files_are_trackable_and_not_lfs_managed() -> None:
    paths = (
        "reports/presentation/index.html",
        "reports/presentation/manifest.json",
        "reports/presentation/assets/ui/presentation.css",
        "reports/presentation/assets/ui/presentation.js",
    )

    ignore_result = _git("check-ignore", "--no-index", "--", *paths)
    assert ignore_result.returncode == 1
    assert ignore_result.stdout == ""
    assert ignore_result.stderr == ""

    attr_result = _git("check-attr", "filter", "--", *paths)
    assert attr_result.returncode == 0
    assert attr_result.stdout.splitlines() == [f"{path}: filter: unspecified" for path in paths]
    assert attr_result.stderr == ""


def test_large_assets_resolve_to_lfs_and_interrupted_writes_are_ignored() -> None:
    lfs_paths = (
        "reports/presentation/assets/notebooks/example.js",
        "reports/presentation/assets/vendor/plotly.min.js",
    )

    attr_result = _git("check-attr", "filter", "--", *lfs_paths)
    assert attr_result.returncode == 0
    assert attr_result.stdout.splitlines() == [f"{path}: filter: lfs" for path in lfs_paths]
    assert attr_result.stderr == ""

    temporary_path = "reports/presentation/notebooks/.tmp-export.html"
    ignore_result = _git("check-ignore", "--no-index", "--", temporary_path)
    assert ignore_result.returncode == 0
    assert ignore_result.stdout == f"{temporary_path}\n"
    assert ignore_result.stderr == ""
