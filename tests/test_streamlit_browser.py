"""Headless-browser checks for the live Streamlit app.

AppTest executes the Streamlit script but never the frontend, and it provably did
NOT catch the previous blank-map failure: it reported zero exceptions while a JS
`ReferenceError: feature_group_<hash> is not defined` had blanked the Overview map
completely. These tests are the only thing that can catch that class of bug, so they
gate any change to how folium layers reach the browser.
"""

import socket
import subprocess
import time
from collections.abc import Iterator
from typing import Any

import pytest

APP_ENTRY = "app/streamlit_app.py"
STARTUP_TIMEOUT_S = 180


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.fixture(scope="module")
def streamlit_server() -> Iterator[str]:
    """Run the real app on a random port and yield its base URL."""
    port = _free_port()
    process = subprocess.Popen(  # noqa: S603
        [
            "uv",
            "run",
            "streamlit",
            "run",
            APP_ENTRY,
            "--server.port",
            str(port),
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                output = process.stdout.read().decode() if process.stdout else ""
                pytest.fail(f"Streamlit exited before serving:\n{output}")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    break
            except OSError:
                time.sleep(1)
        else:
            pytest.fail(f"Streamlit did not start within {STARTUP_TIMEOUT_S}s")
        yield f"http://127.0.0.1:{port}"
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()


def _open_app(browser: Any, url: str) -> tuple[Any, Any, list[str]]:
    context = browser.new_context(viewport={"width": 1440, "height": 900})
    page = context.new_page()
    page_errors: list[str] = []
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    page.goto(url, wait_until="load")
    return context, page, page_errors


def _wait_for_leaflet_paths(page: Any, minimum: int) -> int:
    """Wait until at least `minimum` Leaflet vector paths have rendered."""
    page.wait_for_function(
        "min => {"
        "  const frames = Array.from(document.querySelectorAll('iframe'));"
        "  return frames.some(frame => {"
        "    const doc = frame.contentDocument;"
        "    return doc && doc.querySelectorAll('path.leaflet-interactive').length >= min;"
        "  });"
        "}",
        arg=minimum,
        timeout=120_000,
    )
    return int(
        page.evaluate(
            "() => Math.max(0, ...Array.from(document.querySelectorAll('iframe'))"
            ".map(frame => frame.contentDocument"
            " ? frame.contentDocument.querySelectorAll('path.leaflet-interactive').length"
            " : 0))"
        )
    )


def _open_page_by_name(page: Any, name: str) -> None:
    page.get_by_role("link", name=name).click()
    page.wait_for_timeout(2_000)


@pytest.mark.browser
def test_overview_severity_map_renders_without_js_errors(
    chromium_browser: Any, streamlit_server: str
) -> None:
    context, page, page_errors = _open_app(chromium_browser, streamlit_server)
    try:
        # 4857 cells are drawn; require a large fraction to have rendered so a
        # partially-broken layer set cannot pass.
        rendered = _wait_for_leaflet_paths(page, 4_000)
        assert rendered >= 4_000
        assert page_errors == [], f"JS errors on Overview: {page_errors}"
    finally:
        context.close()


@pytest.mark.browser
def test_overview_layer_control_lists_every_risk_band(
    chromium_browser: Any, streamlit_server: str
) -> None:
    context, page, page_errors = _open_app(chromium_browser, streamlit_server)
    try:
        _wait_for_leaflet_paths(page, 4_000)
        labels = page.evaluate(
            "() => Array.from(document.querySelectorAll('iframe'))"
            ".flatMap(frame => frame.contentDocument"
            " ? Array.from(frame.contentDocument.querySelectorAll("
            "'.leaflet-control-layers-overlays label')).map(node => node.textContent.trim())"
            " : [])"
        )
        joined = " | ".join(labels)
        for band_label in (
            "Well below average (<0.75x)",
            "Around average (0.75-1.1x)",
            "Elevated (1.1-1.5x)",
            "High (1.5-2x)",
            "Very high (>=2x)",
        ):
            assert band_label in joined, f"missing layer toggle {band_label!r} in {joined!r}"
        assert page_errors == []
    finally:
        context.close()


@pytest.mark.browser
def test_overview_map_survives_navigating_away_and_back(
    chromium_browser: Any, streamlit_server: str
) -> None:
    # The FeatureGroups are cached with st.cache_resource. If streamlit-folium
    # mutates them while rendering, the second render would break - this is the test
    # that catches it. On failure, drop @st.cache_resource from
    # build_severity_feature_groups and re-run.
    context, page, page_errors = _open_app(chromium_browser, streamlit_server)
    try:
        _wait_for_leaflet_paths(page, 4_000)
        _open_page_by_name(page, "Model Comparison")
        _open_page_by_name(page, "Overview")
        rendered = _wait_for_leaflet_paths(page, 4_000)
        assert rendered >= 4_000
        assert page_errors == [], f"JS errors after re-navigation: {page_errors}"
    finally:
        context.close()


@pytest.mark.browser
def test_risk_predictor_places_the_marker_on_a_single_click(
    chromium_browser: Any, streamlit_server: str
) -> None:
    context, page, page_errors = _open_app(chromium_browser, streamlit_server)
    try:
        _open_page_by_name(page, "Risk Predictor")
        map_frame = page.frame_locator("iframe").first
        container = map_frame.locator(".leaflet-container")
        container.wait_for(timeout=120_000)

        before = page.get_by_text("Selected: lat").inner_text()
        box = container.bounding_box()
        # Click left-of-centre and above centre: still inside Germany's bbox at
        # zoom 6 centred on [51.1657, 10.4515], but clearly away from the default
        # marker so the readout must change.
        container.click(position={"x": box["width"] * 0.45, "y": box["height"] * 0.4})
        page.wait_for_timeout(8_000)
        after = page.get_by_text("Selected: lat").inner_text()

        # ONE click must move the point. Before the fix this required two.
        assert after != before, f"coordinate readout unchanged after one click: {before!r}"
        assert page_errors == [], f"JS errors on Risk Predictor: {page_errors}"
    finally:
        context.close()
