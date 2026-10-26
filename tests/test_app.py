from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import harvestguard_ai_view


ROOT = Path(__file__).resolve().parents[1]


def _select(app, label, value):
    widget = next(item for item in app.selectbox if item.label == label)
    widget.select(value).run(timeout=40)
    assert not app.exception, [error.message for error in app.exception]


def _text(app):
    return "\n".join(item.value for item in app.markdown)


@pytest.mark.parametrize("entrypoint", ["harvestguard_app.py", "dashboard.py"])
def test_views_follow_filters_and_work_without_ai(monkeypatch, entrypoint):
    monkeypatch.setattr(harvestguard_ai_view, "_setting", lambda name, default="": "" if name == "GROQ_API_KEY" else default)
    app = AppTest.from_file(str(ROOT / entrypoint)).run(timeout=40)
    assert not app.exception, [error.message for error in app.exception]
    assert not app.error
    assert app.radio(key="page").options == ["Map", "Insights", "Ask HarvestGuard"]
    assert len(app.get("plotly_chart")) >= 1
    assert "Nyagatare" in _text(app)
    assert "64,545.97 kg" in _text(app)
    assert "0.79%" in _text(app)
    assert any("HarvestGuard loss model" == item.value for item in app.caption)
    assert not any("Quantities cover sampled profiles only" in item.value for item in app.info)
    assert "Metadata and Methodology" in [item.label for item in app.expander]
    assert "Readiness checks" not in [item.label for item in app.expander]
    assert "Colour shows relative rank" not in "\n".join(item.value for item in app.caption)
    validation = next(frame.value for frame in app.dataframe if "Evaluation" in frame.value.columns)
    exported = pd.read_csv(ROOT / "model_v3_candidate" / "external_validation_2025.csv")
    candidate = exported.loc[(exported.scope == "2025 all") & (exported.model == "V3 locked candidate")].iloc[0]
    displayed = validation.loc[validation.Evaluation == "2025 retrospective validation"].iloc[0]
    assert displayed["District rate error (percentage points)"] == candidate.district_mae_pp
    assert not app.warning
    assert "Predictive importance" in _text(app)
    assert [item.label for item in app.tabs] == ["Data sources", "Methodology", "Limitations", "Project & AI"]
    next(button for button in app.button if button.label == "See insights →").click().run(timeout=40)
    assert not app.exception
    assert app.radio(key="page").value == "Insights"
    assert any(button.label == "Download explanation" for button in app.get("download_button"))
    assert "### Model findings" in _text(app)
    assert "### Where to intervene first" in _text(app)
    assert "What to assess next" not in _text(app)
    assert "AI explanation & evidence" in [item.label for item in app.expander]
    _select(app, "District", "Kamonyi")
    assert "Kamonyi: from insight to action" in _text(app)
    _select(app, "Crop", "Tomato")
    assert "Recommended actions" in _text(app)
    assert "protective, ventilated packaging" not in _text(app)
    assert "hermetic storage" not in _text(app)
    _select(app, "Crop", "Cassava")
    assert "Recommended actions" in _text(app)
    _select(app, "Season", "C")
    assert "no reported positive-loss" in _text(app)
    _select(app, "Season", "A")
    _select(app, "Crop", "Maize")
    assert not any("no reported positive-loss" in item.value for item in app.info)
    app.radio(key="page").set_value("Map").run(timeout=40)
    assert not app.exception
    assert "Where should we look first?" in _text(app)
