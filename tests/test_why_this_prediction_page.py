from streamlit.testing.v1 import AppTest


def test_why_this_prediction_shows_readable_labels_after_a_prediction():
    at = AppTest.from_file("app/pages/risk_predictor.py", default_timeout=30)
    at.run()
    submit_buttons = [b for b in at.button if "Predict KSI risk" in b.label]
    submit_buttons[0].click().run()

    at2 = AppTest.from_file("app/pages/why_this_prediction.py", default_timeout=30)
    at2.session_state["last_prediction"] = at.session_state["last_prediction"]
    at2.run()
    assert not at2.exception

    table_values = [t.value for t in at2.table]
    assert len(table_values) == 1
    feature_column = table_values[0]["feature"].tolist()
    assert "Wind Speed (m/s)" in feature_column
    assert "UART" not in feature_column  # raw code name should not appear, only the label
