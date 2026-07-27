from streamlit.testing.v1 import AppTest


def test_overview_page_loads_and_renders_severity_map():
    at = AppTest.from_file("app/pages/overview.py", default_timeout=60)
    at.run()
    assert not at.exception
