"""Smoke test for the Streamlit page. Never presses Start, that would open the webcam."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parents[1] / "app.py")


def test_app_loads_without_errors():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value == "Latency Limbo"


def test_model_picker_has_three_models_plus_green():
    at = AppTest.from_file(APP, default_timeout=60).run()
    picker = [s for s in at.sidebar.selectbox if s.label == "Hand detector model"][0]
    assert len(picker.options) == 4
    assert picker.options[-1] == "Green sticker (colour tracking)"


def test_results_empty_at_start():
    at = AppTest.from_file(APP, default_timeout=60).run()
    captions = [c.value for c in at.caption]
    assert "Play a round to see results here." in captions
