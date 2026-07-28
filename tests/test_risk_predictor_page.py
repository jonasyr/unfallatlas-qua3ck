from streamlit.testing.v1 import AppTest


def test_risk_predictor_page_loads():
    at = AppTest.from_file("app/pages/risk_predictor.py", default_timeout=180)
    at.run()
    assert not at.exception


def test_risk_predictor_page_does_not_excuse_the_reload_pause():
    # The old copy told users the click lag was expected behaviour. It was a
    # script-ordering bug, now fixed, so the apology must be gone.
    at = AppTest.from_file("app/pages/risk_predictor.py", default_timeout=180)
    at.run()
    assert not at.exception
    captions = " ".join(element.value for element in at.caption)
    assert "not a freeze" not in captions


def test_risk_predictor_form_submits_a_prediction():
    at = AppTest.from_file("app/pages/risk_predictor.py", default_timeout=180)
    at.run()
    at.button[-1].click().run()
    assert not at.exception
