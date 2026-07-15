from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from shutil import which
from typing import Any

import pytest

from unfallatlas.presentation.metadata import build_export_metadata
from unfallatlas.presentation.rendering import render_notebook
from unfallatlas.presentation.validation import read_and_validate_notebook

REPO_ROOT = Path(__file__).parents[2]
GALLERY_NOTEBOOK = Path(__file__).parent / "fixtures" / "gallery.ipynb"
SCREENSHOT_ROOT = Path("/tmp/unfallatlas-presentation-verification")


@pytest.fixture(scope="session")
def gallery_html(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Export the saved gallery state without executing any notebook cell."""
    output_root = tmp_path_factory.mktemp("presentation-browser")
    analysis = read_and_validate_notebook(GALLERY_NOTEBOOK, REPO_ROOT)
    assert analysis.counts.markdown == 3
    assert analysis.counts.code == 7
    assert analysis.counts.executed_without_output == 1
    assert analysis.findings

    metadata = build_export_metadata(
        REPO_ROOT,
        now=datetime(2026, 7, 15, 8, 30, tzinfo=UTC),
    )
    result = render_notebook(analysis, metadata, output_root, repo_root=REPO_ROOT)

    assert result.error is None
    assert result.destination.is_file()
    return result.destination


@pytest.fixture(scope="session")
def chromium_browser() -> Iterator[Any]:
    playwright = pytest.importorskip(
        "playwright.sync_api",
        reason="install the presentation-test extra to run opt-in browser checks",
    )
    with playwright.sync_playwright() as runtime:
        launch_options: dict[str, object] = {"headless": True}
        if not Path(runtime.chromium.executable_path).is_file():
            system_chromium = next(
                (
                    executable
                    for name in ("chromium", "chromium-browser", "google-chrome", "chrome")
                    if (executable := which(name)) is not None
                ),
                None,
            )
            if system_chromium is None:
                pytest.fail(
                    "No Playwright-managed or system Chromium executable is available. "
                    "Run `uv run playwright install chromium`."
                )
            launch_options["executable_path"] = system_chromium
        browser = runtime.chromium.launch(**launch_options)
        yield browser
        browser.close()


def _open_gallery(
    browser: Any,
    gallery_html: Path,
    *,
    width: int,
    height: int,
    reduced_motion: str = "no-preference",
) -> tuple[Any, Any, list[str], list[str], list[str]]:
    context = browser.new_context(
        viewport={"width": width, "height": height},
        reduced_motion=reduced_motion,
    )
    page = context.new_page()
    external_requests: list[str] = []
    console_errors: list[str] = []
    page_errors: list[str] = []
    page.on(
        "request",
        lambda request: (
            external_requests.append(request.url)
            if request.url.startswith(("http://", "https://"))
            else None
        ),
    )
    page.on(
        "console",
        lambda message: console_errors.append(message.text) if message.type == "error" else None,
    )
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    page.goto(gallery_html.resolve().as_uri(), wait_until="load")
    return context, page, external_requests, console_errors, page_errors


def _assert_clean_runtime(
    external_requests: list[str], console_errors: list[str], page_errors: list[str]
) -> None:
    assert external_requests == []
    assert console_errors == []
    assert page_errors == []


def test_gallery_fixture_exports_saved_outputs_without_execution(gallery_html: Path) -> None:
    html = gallery_html.read_text(encoding="utf-8")

    assert "Repräsentative Offline-Galerie" in html
    assert "Gespeicherter Textoutput: 42" in html
    assert "Breite Ergebnistabelle mit 200 Zeilen" in html
    assert "synthetic-validation-warning" in GALLERY_NOTEBOOK.read_text(encoding="utf-8")
    assert "Code cell was executed and legitimately stored no output." in html


@pytest.mark.parametrize(
    ("name", "width", "height"),
    [("desktop", 1440, 900), ("laptop", 1366, 768), ("mobile", 390, 844)],
)
@pytest.mark.browser
def test_gallery_layout_is_usable_at_supported_viewports(
    chromium_browser: Any,
    gallery_html: Path,
    name: str,
    width: int,
    height: int,
) -> None:
    context, page, requests, console_errors, page_errors = _open_gallery(
        chromium_browser, gallery_html, width=width, height=height
    )
    try:
        toc = page.locator("#presentation-toc")
        main = page.locator("#notebook-content")
        if name == "mobile":
            trigger = page.locator('[data-action="toggle-toc"]')
            assert trigger.is_visible()
            assert not toc.is_visible()
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")

            toolbar = page.locator(".presentation-toolbar")
            assert toolbar.evaluate("node => node.scrollWidth > node.clientWidth")
            toolbar.evaluate("node => { node.scrollLeft = node.scrollWidth; }")
            last_control = toolbar.locator("button").last
            assert last_control.is_visible()
            assert last_control.evaluate(
                "node => { const button = node.getBoundingClientRect(); "
                "const toolbar = node.parentElement.getBoundingClientRect(); "
                "return button.left >= toolbar.left && button.right <= toolbar.right + 1; }"
            )

            trigger.click()
            assert toc.is_visible()
            toc_box = toc.bounding_box()
            assert toc_box is not None
            assert toc_box["x"] >= 0
            assert toc_box["x"] + toc_box["width"] <= width
            page.locator('[data-action="close-toc"]').click()
            assert not toc.is_visible()
        else:
            assert toc.is_visible()
            toc_box = toc.bounding_box()
            main_box = main.bounding_box()
            assert toc_box is not None and main_box is not None
            assert toc_box["x"] + toc_box["width"] <= main_box["x"]

        SCREENSHOT_ROOT.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=SCREENSHOT_ROOT / f"gallery-{name}.png", full_page=True)
        _assert_clean_runtime(requests, console_errors, page_errors)
    finally:
        context.close()


@pytest.mark.browser
def test_keyboard_controls_and_code_output_toggles_keep_aria_in_sync(
    chromium_browser: Any, gallery_html: Path
) -> None:
    context, page, requests, console_errors, page_errors = _open_gallery(
        chromium_browser, gallery_html, width=1366, height=768
    )
    try:
        visible_actions = page.locator(".presentation-toolbar button:visible")
        expected_actions = set(
            visible_actions.evaluate_all("nodes => nodes.map(node => node.dataset.action)")
        )
        reached: set[str] = set()
        for _ in range(20):
            page.keyboard.press("Tab")
            focused = page.locator(":focus")
            action = focused.get_attribute("data-action")
            if action:
                reached.add(action)
                assert focused.evaluate("node => node.matches(':focus-visible')")
                assert focused.evaluate("node => getComputedStyle(node).outlineStyle") != "none"
            if expected_actions <= reached:
                break
        assert expected_actions <= reached

        initially_offscreen_plot = page.locator(".plotly-output")
        output_details = page.locator("details.output-cell")
        assert page.locator("details.output-cell:not([open])").count() == 0
        assert (
            page.locator("details.output-cell > summary[aria-expanded='true']").count()
            == output_details.count()
        )
        assert initially_offscreen_plot.get_attribute("data-loaded") != "true"
        page.locator('[data-action="show-all-output"]').click()
        page.wait_for_function(
            "() => document.querySelector('.plotly-output').dataset.loaded === 'true'",
            timeout=15_000,
        )

        first_code = page.locator("details.code-cell").first
        first_output = page.locator("details.output-cell").first
        first_code.locator(":scope > summary").click()
        first_output.locator(":scope > summary").click()
        assert first_code.get_attribute("open") is not None
        assert first_code.locator(":scope > summary").get_attribute("aria-expanded") == "true"
        assert first_output.get_attribute("open") is None
        assert first_output.locator(":scope > summary").get_attribute("aria-expanded") == "false"

        page.locator('[data-action="show-all-code"]').click()
        page.wait_for_function(
            "() => [...document.querySelectorAll('details.code-cell')]"
            ".every(details => details.open && "
            "details.querySelector(':scope > summary').getAttribute('aria-expanded') === 'true')"
        )
        assert page.locator("details.code-cell:not([open])").count() == 0
        assert (
            page.locator("details.code-cell > summary[aria-expanded='true']").count()
            == page.locator("details.code-cell").count()
        )

        page.locator('[data-action="hide-all-code"]').click()
        page.wait_for_function(
            "() => [...document.querySelectorAll('details.code-cell')]"
            ".every(details => !details.open && "
            "details.querySelector(':scope > summary').getAttribute('aria-expanded') === 'false')"
        )
        assert page.locator("details.code-cell[open]").count() == 0
        assert (
            page.locator("details.code-cell > summary[aria-expanded='false']").count()
            == page.locator("details.code-cell").count()
        )

        page.locator('[data-action="hide-all-output"]').click()
        page.wait_for_function(
            "() => [...document.querySelectorAll('details.output-cell')]"
            ".every(details => !details.open && "
            "details.querySelector(':scope > summary').getAttribute('aria-expanded') === 'false')"
        )
        assert page.locator("details.output-cell[open]").count() == 0
        assert (
            page.locator("details.output-cell > summary[aria-expanded='false']").count()
            == page.locator("details.output-cell").count()
        )

        page.locator('[data-action="show-all-output"]').click()
        page.wait_for_function(
            "() => [...document.querySelectorAll('details.output-cell')]"
            ".every(details => details.open && "
            "details.querySelector(':scope > summary').getAttribute('aria-expanded') === 'true')"
        )
        assert page.locator("details.output-cell:not([open])").count() == 0
        assert (
            page.locator("details.output-cell > summary[aria-expanded='true']").count()
            == page.locator("details.output-cell").count()
        )
        page.wait_for_selector(".plotly-output.js-plotly-plot", timeout=15_000)
        _assert_clean_runtime(requests, console_errors, page_errors)
    finally:
        context.close()


@pytest.mark.browser
def test_detail_state_survives_reload_for_the_same_snapshot(
    chromium_browser: Any, gallery_html: Path
) -> None:
    context, page, requests, console_errors, page_errors = _open_gallery(
        chromium_browser, gallery_html, width=1366, height=768
    )
    try:
        page.locator('[data-action="show-all-code"]').click()
        page.wait_for_function(
            "() => [...document.querySelectorAll('details.code-cell > summary')]"
            ".every(summary => summary.getAttribute('aria-expanded') === 'true')"
        )
        first_code_summary = page.locator("details.code-cell").first.locator(":scope > summary")
        first_output_summary = page.locator("details.output-cell").first.locator(":scope > summary")
        first_code_index = page.locator("details.code-cell").first.evaluate(
            "node => [...document.querySelectorAll('details')].indexOf(node)"
        )
        first_output_index = page.locator("details.output-cell").first.evaluate(
            "node => [...document.querySelectorAll('details')].indexOf(node)"
        )
        first_code_summary.click()
        first_output_summary.click()
        page.wait_for_function(
            "() => document.querySelector('details.code-cell > summary')"
            ".getAttribute('aria-expanded') === 'false'"
        )
        page.wait_for_function(
            "() => document.querySelector('details.output-cell > summary')"
            ".getAttribute('aria-expanded') === 'false'"
        )
        stored_states = page.evaluate(
            "Object.fromEntries(Object.entries(sessionStorage)"
            ".filter(([key]) => key.startsWith('unfallatlas-presentation:')))"
        )
        assert (
            sum(
                key.endswith(f":details:code:{first_code_index}") and value == "closed"
                for key, value in stored_states.items()
            )
            == 1
        )
        assert (
            sum(
                key.endswith(f":details:output:{first_output_index}") and value == "closed"
                for key, value in stored_states.items()
            )
            == 1
        )
        page.reload(wait_until="load")

        assert page.locator("details.code-cell").first.get_attribute("open") is None
        assert page.locator("details.code-cell").nth(1).get_attribute("open") is not None
        assert page.locator("details.output-cell").first.get_attribute("open") is None
        assert (
            page.locator("details.code-cell")
            .first.locator(":scope > summary")
            .get_attribute("aria-expanded")
            == "false"
        )
        _assert_clean_runtime(requests, console_errors, page_errors)
    finally:
        context.close()


@pytest.mark.browser
def test_large_table_and_log_can_be_scrolled_and_expanded(
    chromium_browser: Any, gallery_html: Path
) -> None:
    context, page, requests, console_errors, page_errors = _open_gallery(
        chromium_browser, gallery_html, width=390, height=844
    )
    try:
        table = page.locator(".table-scroll")
        dimensions = table.evaluate(
            "node => ({scrollWidth: node.scrollWidth, clientWidth: node.clientWidth, "
            "scrollHeight: node.scrollHeight, clientHeight: node.clientHeight})"
        )
        assert dimensions["scrollWidth"] > dimensions["clientWidth"]
        assert dimensions["scrollHeight"] > dimensions["clientHeight"]

        table_button = page.locator(f'button[aria-controls="{table.get_attribute("id")}"]')
        assert table_button.is_visible()
        assert table_button.get_attribute("aria-expanded") == "false"
        table_button.click()
        assert table.get_attribute("class").split().count("is-expanded") == 1
        assert table_button.get_attribute("aria-expanded") == "true"
        assert table.evaluate("node => getComputedStyle(node).maxHeight") == "none"
        _assert_clean_runtime(requests, console_errors, page_errors)
    finally:
        context.close()


@pytest.mark.browser
def test_plotly_lazy_loads_from_local_assets_without_runtime_errors(
    chromium_browser: Any, gallery_html: Path
) -> None:
    context, page, requests, console_errors, page_errors = _open_gallery(
        chromium_browser, gallery_html, width=390, height=844
    )
    try:
        plot = page.locator(".plotly-output")
        assert plot.get_attribute("data-loaded") != "true"
        assert page.locator(".plotly-output.js-plotly-plot").count() == 0
        dynamic_script_sources = page.locator("script[src]").evaluate_all(
            "scripts => scripts.map(script => script.getAttribute('src'))"
        )
        assert (
            page.locator("body").get_attribute("data-plotly-runtime") not in dynamic_script_sources
        )
        assert plot.get_attribute("data-asset") not in dynamic_script_sources

        plot.scroll_into_view_if_needed()
        page.wait_for_selector(".plotly-output.js-plotly-plot", timeout=15_000)
        assert plot.get_attribute("data-loaded") == "true"
        metrics = plot.evaluate(
            "node => ({clientWidth: node.clientWidth, scrollWidth: node.scrollWidth, "
            "overflowX: getComputedStyle(node).overflowX, "
            "left: node.getBoundingClientRect().left, right: node.getBoundingClientRect().right})"
        )
        assert metrics["left"] >= 0
        assert metrics["right"] <= 390
        assert metrics["scrollWidth"] <= metrics["clientWidth"] or metrics["overflowX"] in {
            "auto",
            "scroll",
        }
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        SCREENSHOT_ROOT.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=SCREENSHOT_ROOT / "gallery-mobile-plotly.png", full_page=True)
        _assert_clean_runtime(requests, console_errors, page_errors)
    finally:
        context.close()


@pytest.mark.browser
def test_reduced_motion_and_print_media_preserve_readable_static_content(
    chromium_browser: Any, gallery_html: Path
) -> None:
    context, page, requests, console_errors, page_errors = _open_gallery(
        chromium_browser,
        gallery_html,
        width=1366,
        height=768,
        reduced_motion="reduce",
    )
    try:
        duration = page.locator(".notebook-cell").first.evaluate(
            "node => parseFloat(getComputedStyle(node).transitionDuration) || 0"
        )
        assert duration <= 0.001

        page.locator('[data-action="hide-all-code"]').click()
        page.locator('[data-action="hide-all-output"]').click()
        page.evaluate("window.dispatchEvent(new Event('beforeprint'))")
        page.emulate_media(media="print")
        assert page.locator("details:not([open])").count() == 0
        for selector in (
            ".presentation-toolbar",
            "#presentation-toc",
            ".back-to-top",
            ".skip-link",
        ):
            assert (
                page.locator(selector).evaluate("node => getComputedStyle(node).display") == "none"
            )
        assert page.locator("button:visible").count() == 0
        assert page.locator("summary:visible").count() == 0
        assert (
            page.locator("details.code-cell .code-content").first.evaluate(
                "node => getComputedStyle(node).display"
            )
            == "block"
        )
        assert page.locator("details.code-cell .code-content").first.is_visible()
        assert page.locator("details.output-cell .output-content").first.is_visible()
        assert (
            page.locator("details.output-cell .output-content").first.evaluate(
                "node => getComputedStyle(node).display"
            )
            == "block"
        )
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")

        SCREENSHOT_ROOT.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=SCREENSHOT_ROOT / "gallery-print.png", full_page=True)
        _assert_clean_runtime(requests, console_errors, page_errors)
    finally:
        context.close()
