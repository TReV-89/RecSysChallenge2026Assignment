from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_setup_screen_without_catalog(monkeypatch, tmp_path):
    from music_recommender.config import Settings

    monkeypatch.setattr(Settings, "from_env", classmethod(lambda cls: cls(data_dir=tmp_path)))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=30)
    assert not app.exception
    assert app.title[0].value == "Music for the moment"
    assert "Build the catalog first" in app.info[0].value
    assert app.sidebar.selectbox[0].value == "Full catalog"
