from __future__ import annotations

from streamlit.testing.v1 import AppTest


def test_app_starts_and_sample_inspection_shows_fail():
    app = AppTest.from_file("app.py", default_timeout=30)
    app.run()
    assert not app.exception
    labels = [button.label for button in app.button]
    assert "FOREIGN EDGE Sample 1" in labels
    for button in app.button:
        if button.label == "FOREIGN EDGE Sample 1":
            button.click().run()
            break
    assert not app.exception
    texts = "\n".join(str(md.value) for md in app.markdown)
    assert "FAIL" in texts
    assert "FOREIGN EDGE" in texts
    assert "RESULT" in texts
    assert "COIL SURFACE INSPECTION" in texts
    for button in app.button:
        if button.label == "EDGE RUB Sample 1":
            button.click().run()
            break
    assert not app.exception
    texts = "\n".join(str(md.value) for md in app.markdown)
    assert "EDGE RUB" in texts


def test_analyze_button_appears_when_image_is_queued():
    from tests.conftest import make_uniform_image

    app = AppTest.from_file("app.py", default_timeout=30)
    app.run()
    assert "ANALYZE" in [button.label for button in app.button]
    image = make_uniform_image()
    app.session_state.pending_images = [("coil.jpg", image)]
    app.session_state.last_original = image
    app.session_state.last_source_name = "coil.jpg"
    app.run()
    assert not app.exception
    assert "ANALYZE" in [button.label for button in app.button]


def test_camera_start_does_not_crash_when_unavailable():
    app = AppTest.from_file("app.py", default_timeout=30)
    app.run()
    for button in app.button:
        if button.label == "Start Camera":
            button.click().run()
            break
    assert not app.exception
