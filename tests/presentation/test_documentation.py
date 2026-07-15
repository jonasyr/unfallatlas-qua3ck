import re
from pathlib import Path

import pytest

from unfallatlas.presentation.cli import build_parser

REPO_ROOT = Path(__file__).resolve().parents[2]
GUIDE = REPO_ROOT / "docs" / "presentation-export.md"
README = REPO_ROOT / "README.md"
TEMPLATE = (
    REPO_ROOT / "src" / "unfallatlas" / "presentation" / "templates" / "notebook" / "index.html.j2"
)
GITATTRIBUTES = REPO_ROOT / ".gitattributes"

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


def _section(guide_text: str, heading: str) -> str:
    match = re.search(
        rf"^## {re.escape(heading)}\s*$\n(?P<body>.*?)(?=^## |\Z)",
        guide_text,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"missing guide section: {heading}"
    return match.group("body")


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


def test_widget_and_folium_section_requires_static_fallbacks(guide_text: str) -> None:
    section = _section(guide_text, "Widgets und Karten")
    normalized = section.casefold()

    assert "interaktive jupyter-widgets" in normalized
    assert "nicht unterstützt" in normalized
    assert re.search(r"widget-manager.*nicht veröffentlicht", normalized, flags=re.DOTALL)
    assert re.search(r"folium.*sandbox", normalized, flags=re.DOTALL)
    assert re.search(r"skript.*deaktiviert", normalized, flags=re.DOTALL)
    for fallback in ("png", "svg", "tabelle", "text"):
        assert fallback in normalized


def test_pdf_section_uses_current_print_button_label(guide_text: str) -> None:
    template = TEMPLATE.read_text(encoding="utf-8")
    label = re.search(r'data-action="print">([^<]+)</button>', template)
    assert label is not None

    assert f"„{label.group(1)}“" in _section(guide_text, "PDF")


def test_export_section_describes_open_by_rendered_snapshot_count(guide_text: str) -> None:
    section = " ".join(_section(guide_text, "Export").casefold().split())

    assert "--open" in section
    assert "genau einem snapshot" in section
    assert "tatsächlich gerenderter snapshots" in section
    assert "notebook-html" in section
    assert re.search(r"mehrere snapshots.*index", section, flags=re.DOTALL)


def test_lfs_section_spells_full_presentation_asset_paths(guide_text: str) -> None:
    configured_paths = {
        line.split()[0]
        for line in GITATTRIBUTES.read_text(encoding="utf-8").splitlines()
        if line.startswith("reports/presentation/") and "filter=lfs" in line
    }
    assert configured_paths == {
        "reports/presentation/assets/notebooks/**",
        "reports/presentation/assets/vendor/**",
    }

    section = _section(guide_text, "Git LFS")
    for path in configured_paths:
        assert f"`{path}`" in section


def test_freshness_section_rejects_missing_and_empty_manifests(guide_text: str) -> None:
    section = _section(guide_text, "Freshness").casefold()

    assert re.search(r"manifest.*fehlt.*exit-code `1`", section, flags=re.DOTALL)
    assert re.search(r"manifest.*keine einträge.*exit-code `1`", section, flags=re.DOTALL)
