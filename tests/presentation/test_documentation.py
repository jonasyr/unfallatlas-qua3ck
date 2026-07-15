from pathlib import Path

import pytest

from unfallatlas.presentation.cli import build_parser

REPO_ROOT = Path(__file__).resolve().parents[2]
GUIDE = REPO_ROOT / "docs" / "presentation-export.md"
README = REPO_ROOT / "README.md"

DOCUMENTED_COMMANDS = (
    "uv sync --extra presentation",
    "uv run python scripts/export_notebooks.py --all",
    "uv run python scripts/export_notebooks.py notebooks/02_U_Phase.ipynb",
    "uv run python scripts/export_notebooks.py --all --strict",
    "uv run python scripts/export_notebooks.py --check",
)

REQUIRED_HEADINGS = (
    "Installation",
    "Export",
    "Validation",
    "nbstripout",
    "Offline-Nutzung",
    "PDF",
    "Plotly",
    "Widgets und Karten",
    "Git LFS",
    "Freshness",
    "Platzhalter und WIP",
    "Zukünftige Notebooks",
    "Fehlerbehebung",
    "GitHub Pages",
)


@pytest.fixture
def guide_text() -> str:
    return GUIDE.read_text(encoding="utf-8") if GUIDE.is_file() else ""


def test_readme_links_to_concise_presentation_workflow() -> None:
    readme = README.read_text(encoding="utf-8")

    assert "## Notebook-Präsentationen" in readme
    assert "docs/presentation-export.md" in readme
    assert "uv sync --extra presentation" in readme
    assert "uv run python scripts/export_notebooks.py --all" in readme
    assert "reports/presentation/" in readme


def test_operator_guide_documents_exact_supported_commands(guide_text: str) -> None:
    assert guide_text, "docs/presentation-export.md must provide the operator guide"
    parser_help = build_parser().format_help()

    for command in DOCUMENTED_COMMANDS:
        assert command in guide_text
    for option in ("--all", "--check", "--strict", "--include-placeholders", "--open"):
        assert option in parser_help
        assert option in guide_text


def test_operator_guide_has_required_topics(guide_text: str) -> None:
    for heading in REQUIRED_HEADINGS:
        assert f"## {heading}" in guide_text


def test_operator_guide_records_operational_safety_contracts(guide_text: str) -> None:
    required_phrases = (
        "GitHub Actions",
        "workflow_dispatch",
        "keine der vorgeschlagenen Workflow-Vorlagen",
        "file://",
        "OpenStreetMap",
        "manifest.json",
        "source_sha256",
        "git lfs pull",
        "vollständigen Ordner",
    )
    for phrase in required_phrases:
        assert phrase in guide_text
