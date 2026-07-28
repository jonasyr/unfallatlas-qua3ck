from streamlit.testing.v1 import AppTest


def test_overview_page_loads_and_renders_severity_map():
    at = AppTest.from_file("app/pages/overview.py", default_timeout=120)
    at.run()
    assert not at.exception


def test_overview_page_renders_the_relative_risk_legend():
    at = AppTest.from_file("app/pages/overview.py", default_timeout=120)
    at.run()
    assert not at.exception
    rendered = " ".join(element.value for element in at.markdown)
    # The legend must name every band and state the national baseline, so a reader
    # can tell what "2x" is relative to.
    assert "Well below average (<0.75x)" in rendered
    assert "Very high (>=2x)" in rendered
    assert "18.9%" in rendered
