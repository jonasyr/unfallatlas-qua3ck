import re
import shlex
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
DISCLOSURE = REPO_ROOT / "docs" / "AI TOOL DISCLOSURE.md"
C_PROMPTS = REPO_ROOT / "docs" / "prompts" / "04_prompts_phase_c.md"

INSTALL_COMMAND = "uv sync --extra presentation"
EXPORT_COMMANDS = (
    (
        "uv run python scripts/export_notebooks.py --all",
        {"all": True, "check": False, "strict": False, "notebooks": []},
    ),
    (
        "uv run python scripts/export_notebooks.py notebooks/02_U_Phase.ipynb",
        {
            "all": False,
            "check": False,
            "strict": False,
            "notebooks": ["notebooks/02_U_Phase.ipynb"],
        },
    ),
    (
        "uv run python scripts/export_notebooks.py --all --strict",
        {"all": True, "check": False, "strict": True, "notebooks": []},
    ),
    (
        "uv run python scripts/export_notebooks.py --check",
        {"all": False, "check": True, "strict": False, "notebooks": []},
    ),
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
    assert INSTALL_COMMAND in guide_text

    parser = build_parser()
    parser_help = parser.format_help()

    for command, expected in EXPORT_COMMANDS:
        assert command in guide_text
        tokens = shlex.split(command, posix=True)
        assert tokens[:4] == ["uv", "run", "python", "scripts/export_notebooks.py"]
        arguments = parser.parse_args(tokens[4:])
        for name, value in expected.items():
            assert getattr(arguments, name) == value
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


def test_widget_state_finding_is_conditional_on_missing_static_fallback(
    guide_text: str,
) -> None:
    validation = _section(guide_text, "Validation").casefold()
    widget_row = next(
        line for line in validation.splitlines() if line.startswith("| `widget_state_missing`")
    )
    widgets = " ".join(_section(guide_text, "Widgets und Karten").casefold().split())

    assert re.search(r"kein\w* unterstützt\w* statisch\w* fallback", widget_row)
    assert re.search(
        r"widget-zustand fehlt.*kein unterstützter statischer fallback.*widget_state_missing",
        widgets,
    )
    assert re.search(
        r"mit.*html-, bild- oder text-fallback.*widget-mime.*übersprungen.*ohne diesen befund",
        widgets,
    )
    assert "`widget_unsupported`" in validation
    assert re.search(r"widget_unsupported.*statisch.*fallback", validation, flags=re.DOTALL)


def test_placeholder_and_wip_section_covers_empty_and_error_notebooks(
    guide_text: str,
) -> None:
    section = " ".join(_section(guide_text, "Platzhalter und WIP").casefold().split())

    assert re.search(r"leere notebook.*placeholder", section)
    assert re.search(r"error-output.*wip", section)


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


def test_cross_phase_refactor_is_disclosed_and_linked() -> None:
    disclosure = DISCLOSURE.read_text(encoding="utf-8")
    prompt_record = C_PROMPTS.read_text(encoding="utf-8")

    assert "2026-07-23-presentation-notebook-refactor-design.md" in disclosure
    assert "2026-07-23-presentation-notebook-refactor.md" in disclosure
    assert "interactive Plotly" in disclosure
    assert "Comprehensive project review and refactor" in prompt_record
    assert "English-only" in prompt_record
    for notebook_name in (
        "notebooks/01_Q_Phase.ipynb",
        "notebooks/02_U_Phase.ipynb",
        "notebooks/03_A3_Phase.ipynb",
        "notebooks/04_C_Phase.ipynb",
    ):
        assert notebook_name in prompt_record
    assert "all ten persisted candidates" in prompt_record
    assert "Random Forest, XGBoost, LightGBM, and CatBoost" in prompt_record
    assert "one confirmation" in prompt_record
    assert "2026-07-23-presentation-notebook-refactor-design.md" in prompt_record
    assert "2026-07-23-presentation-notebook-refactor.md" in prompt_record
    assert "matching `.py` mirrors" in prompt_record


def _markdown_prose(document: str) -> str:
    prose_lines: list[str] = []
    in_fence = False

    for line in document.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        line = re.sub(r"^\s*(?:>\s*)+", "", line)
        if line.startswith(("    ", "\t")):
            continue
        line = re.sub(r"`[^`]*`", "", line)
        line = re.sub(r"^\s*[-*+]\s+", "", line)
        prose_lines.append(line)

    return "\n".join(prose_lines)


def test_markdown_prose_ignores_non_prose_dash_syntax() -> None:
    markdown = """\
- top-level item
  - nested item
> - quoted item
`a - b`
    a - b
```
a - b
```
"""

    assert " - " not in _markdown_prose(markdown)


@pytest.mark.parametrize("document", [DISCLOSURE, C_PROMPTS])
def test_updated_ai_provenance_markdown_avoids_sentence_dash_punctuation(
    document: Path,
) -> None:
    text = _markdown_prose(document.read_text(encoding="utf-8"))

    assert "—" not in text
    assert "–" not in text
    assert not re.search(r"[ \t]-[ \t]", text)
