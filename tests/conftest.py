from collections.abc import Iterator
from pathlib import Path
from shutil import which
from typing import Any

import pytest


@pytest.fixture(scope="session")
def chromium_browser() -> Iterator[Any]:
    playwright = pytest.importorskip(
        "playwright.sync_api",
        reason="install the presentation-test extra to run opt-in browser checks",
    )
    with playwright.sync_playwright() as runtime:
        launch_options: dict[str, object] = {"headless": True, "args": ["--no-sandbox"]}
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
