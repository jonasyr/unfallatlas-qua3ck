from streamlit.testing.v1 import AppTest


def test_risk_predictor_page_loads_with_default_location_and_can_predict():
    at = AppTest.from_file("app/pages/risk_predictor.py", default_timeout=30)
    at.run()
    assert not at.exception

    submit_buttons = [b for b in at.button if "Predict KSI risk" in b.label]
    assert len(submit_buttons) == 1
    submit_buttons[0].click().run()
    assert not at.exception
    assert any("Prediction:" in md.value for md in at.markdown)
