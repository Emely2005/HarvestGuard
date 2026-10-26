"""The dashboard and briefs must keep quantity priorities through every layer."""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import harvestguard_ai_view
import harvestguard_data
from harvestguard_ai import build_evidence_packet, request_fingerprint
from harvestguard_data import MODEL_FOLDER, PREDICTIONS_FILE, REQUIRED_FILES, load_model_data, rank_districts
from harvestguard_decisions import build_brief, brief_to_markdown, load_knowledge_base


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def quantity_bundle(tmp_path):
    """Hypothetical test quantities; never overwrite the real saved bundle."""
    bundle = tmp_path
    for filename in REQUIRED_FILES:
        shutil.copyfile(ROOT / MODEL_FOLDER / filename, bundle / filename)
    risk = pd.read_csv(bundle / PREDICTIONS_FILE)
    risk["production_kg"] = 1000.
    risk.loc[risk.district.eq("Nyagatare"), "production_kg"] = 100000.
    for kind in ("predicted", "observed"):
        risk[f"{kind}_loss_kg"] = risk[f"{kind}_loss_pct"] / 100 * risk.production_kg
        risk[f"{kind}_loss_tonnes"] = risk[f"{kind}_loss_kg"] / 1000
    risk["loss_quantity_scope"] = "sample_total"
    risk["priority_rank"] = risk.groupby(["season", "crop_name"]).predicted_loss_kg.rank(method="min", ascending=False).astype(int)
    risk["vulnerability_rank"] = risk.priority_rank
    risk.to_csv(bundle / PREDICTIONS_FILE, index=False)
    path = bundle / "model_v3_metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata.update(production_burden_available=True, ranking_basis="predicted_loss_kg", loss_quantity_scope="sample_total")
    path.write_text(json.dumps(metadata), encoding="utf-8")
    return bundle


def test_lower_rate_larger_loss_keeps_priority_in_brief_and_ai(quantity_bundle):
    risk, metadata, _ = load_model_data(quantity_bundle)
    view = rank_districts(risk.loc[risk.season.eq("A") & risk.crop_name.eq("Maize")])
    assert view.iloc[0].district == "Nyagatare"
    assert view.iloc[0].predicted_loss_pct < view.predicted_loss_pct.max()
    kb = load_knowledge_base(ROOT / "knowledge_base" / "interventions.json")
    brief = build_brief(view.iloc[0].to_dict(), metadata, kb, district_count=len(view))
    assert brief["expected_loss_kg"] == view.iloc[0].predicted_loss_kg
    assert brief["ranking_basis"] == "predicted_loss_kg"
    assert "sampled profiles only" in brief["interpretation"]
    assert "estimated kg lost" in brief_to_markdown(brief)
    packet = build_evidence_packet(brief, view.to_dict("records"))
    ranking = next(f for f in packet["facts"] if f["id"] == "ranking")
    assert ranking["district_comparison"][0]["district"] == "Nyagatare"
    assert "summed estimated kg lost" in ranking["text"]
    assert "sample_total" in ranking["text"]
    assert packet["display_values"]["loss_kg"].endswith(" kg")
    assert "quantity" in {f["id"] for f in packet["facts"]}
    changed = {**packet, "display_values": {**packet["display_values"], "loss_kg": "999 kg"}}
    assert request_fingerprint(packet, "model", "") != request_fingerprint(changed, "model", "")
    json.dumps(packet, allow_nan=False)
    peers = view.to_dict("records")
    peers[0]["predicted_loss_kg"] += 1
    with pytest.raises(ValueError, match="does not match"):
        build_evidence_packet(brief, peers)


@pytest.mark.parametrize("problem", ["negative", "missing", "partial", "scope", "metadata", "rank", "arithmetic", "tonnes"])
def test_invalid_quantity_exports_are_rejected(quantity_bundle, problem):
    path = quantity_bundle / PREDICTIONS_FILE
    risk = pd.read_csv(path)
    if problem == "negative":
        risk.loc[0, "predicted_loss_kg"] = -1
    elif problem == "missing":
        risk.loc[0, "production_kg"] = np.nan
    elif problem == "partial":
        risk = risk.drop(columns="production_kg")
    elif problem == "scope":
        risk.loc[0, "loss_quantity_scope"] = "survey_expanded"
    elif problem == "metadata":
        metadata_path = quantity_bundle / "model_v3_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["survey_weights_available"] = True
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    elif problem == "rank":
        risk.loc[0, "priority_rank"] = 99
    elif problem == "arithmetic":
        risk.loc[0, "predicted_loss_kg"] += 100
    elif problem == "tonnes":
        risk.loc[0, "predicted_loss_tonnes"] += 100
    risk.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_model_data(quantity_bundle)


def test_dashboard_ranks_and_explains_quantity_without_ai(quantity_bundle, monkeypatch):
    loaded = load_model_data(quantity_bundle)
    monkeypatch.setattr(harvestguard_data, "load_model_data", lambda _: loaded)
    monkeypatch.setattr(harvestguard_data, "dashboard_signature", lambda _: ("quantity-test", str(quantity_bundle)))
    monkeypatch.setattr(harvestguard_ai_view, "_setting", lambda name, default="": "" if name == "GROQ_API_KEY" else default)
    app = AppTest.from_file(str(ROOT / "dashboard.py")).run(timeout=40)
    assert not app.exception
    assert next(m for m in app.metric if m.label == "First in this selection").value == "Nyagatare"
    assert any(m.label == "Highest estimated loss" and "kg" in m.value for m in app.metric)
    table = next(f.value for f in app.dataframe if "predicted_loss_kg" in f.value)
    assert table.iloc[0].District == "Nyagatare"
    assert "loss_quantity_scope" in table
    assert not any("sampled profiles only" in item.value for item in app.info)
    assert "Quantities cover the sampled profiles" in "\n".join(item.value for item in app.markdown)
    app.radio(key="page").set_value("Insights").run(timeout=40)
    assert not app.exception
    assert "by estimated kg lost" in "\n".join(m.value for m in app.markdown)


def test_legacy_rate_view_does_not_claim_quantity_priorities():
    risk, metadata, _ = load_model_data(ROOT / MODEL_FOLDER)
    assert metadata["ranking_basis"] == "predicted_loss_pct"
    assert metadata["loss_quantity_scope"] == "unavailable"
    assert "predicted_loss_kg" not in risk


def test_quantity_ties_share_display_rank():
    view = pd.DataFrame({"district": ["B", "A", "C"], "predicted_loss_kg": [100, 100, 10], "predicted_loss_pct": [1, 20, 50]})
    ranked = rank_districts(view)
    assert ranked.district.tolist() == ["A", "B", "C"]
    assert ranked.display_rank.tolist() == [1, 1, 3]
