import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import harvestguard_ai_view
from harvestguard_ai import build_evidence_packet, request_fingerprint
from harvestguard_data import (
    MODEL_FOLDER, PREDICTIONS_FILE, REQUIRED_FILES, bundle_signature, load_model_data,
)
from harvestguard_decisions import build_brief, load_knowledge_base


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / MODEL_FOLDER


@pytest.fixture
def bundle(tmp_path):
    for filename in REQUIRED_FILES:
        shutil.copyfile(MODEL_DIR / filename, tmp_path / filename)
    return tmp_path


def test_all_v3_predictions_preserve_candidate_identity_and_evidence():
    risk, metadata, readiness = load_model_data(MODEL_DIR)
    exported = pd.read_csv(MODEL_DIR / PREDICTIONS_FILE)
    pd.testing.assert_frame_equal(risk, exported)
    assert len(risk) == 944
    assert risk.district.nunique() == 30
    assert risk.crop_name.nunique() == 15
    assert set(risk.season) == {"A", "B", "C"}
    assert readiness.passed.tolist() == [True, False, False]
    kb = load_knowledge_base(ROOT / "knowledge_base" / "interventions.json")
    for _, group in risk.groupby(["season", "crop_name"]):
        group = group.sort_values(["predicted_loss_pct", "district"], ascending=[False, True]).copy()
        group["display_rank"] = range(1, len(group) + 1)
        for row in group.to_dict("records"):
            brief = build_brief(row, metadata, kb, district_count=len(group))
            assert brief["expected_loss_pct"] == row["predicted_loss_pct"]
            assert brief["model_version"] == "harvestguard-v3-candidate-1"
            assert "previously inspected" in " ".join(brief["limitations"])
            assert "Model V2" not in json.dumps(brief)
            packet = build_evidence_packet(brief, group.to_dict("records"))
            assert packet["prediction_source"]["prediction_file"] == PREDICTIONS_FILE
            json.dumps(packet, allow_nan=False)


@pytest.mark.parametrize("problem", ["missing_file", "duplicate", "invalid_percent", "nonfinite", "missing_column", "empty", "bad_metadata"])
def test_invalid_candidate_bundle_is_rejected(bundle, problem):
    path = bundle / PREDICTIONS_FILE
    risk = pd.read_csv(path)
    if problem == "missing_file":
        (bundle / "model_v3_candidate.joblib").unlink()
    elif problem == "bad_metadata":
        (bundle / "model_v3_metadata.json").write_text("[]", encoding="utf-8")
    else:
        if problem == "duplicate":
            risk = pd.concat([risk, risk.iloc[:1]], ignore_index=True)
        elif problem == "invalid_percent":
            risk.loc[0, "predicted_loss_pct"] = 101
        elif problem == "nonfinite":
            risk.loc[0, "predicted_loss_pct"] = float("nan")
        elif problem == "missing_column":
            risk = risk.drop(columns="support_level")
        elif problem == "empty":
            risk = risk.iloc[:0]
        risk.to_csv(path, index=False)
    with pytest.raises((ValueError, FileNotFoundError)):
        load_model_data(bundle)


def test_replacing_predictions_refreshes_the_data_cache(bundle):
    previous = bundle_signature(bundle)
    path = bundle / PREDICTIONS_FILE
    risk = pd.read_csv(path)
    risk.loc[0, "predicted_loss_pct"] = 0.123456789
    risk.to_csv(path, index=False)
    assert bundle_signature(bundle) != previous
    updated, _, _ = load_model_data(bundle)
    assert updated.iloc[0].predicted_loss_pct == pytest.approx(0.123456789)


def test_ai_uses_v3_evidence_and_does_not_reuse_other_model_versions(monkeypatch):
    packets = []
    original = harvestguard_ai_view.build_evidence_packet

    def capture(*args, **kwargs):
        packet = original(*args, **kwargs)
        packets.append(packet)
        return packet

    monkeypatch.setattr(harvestguard_ai_view, "build_evidence_packet", capture)
    monkeypatch.setattr(harvestguard_ai_view, "_setting", lambda name, default="": "" if name == "GROQ_API_KEY" else default)
    app = AppTest.from_file(str(ROOT / "dashboard.py")).run(timeout=40)
    app.radio(key="page").set_value("Insights").run(timeout=40)
    assert not app.exception
    packet = packets[-1]
    assert packet["prediction_source"]["model_label"] == "Model V3 candidate"
    assert any(fact["kind"] == "global_importance" for fact in packet["facts"])
    assert any(fact["kind"] == "local_model" for fact in packet["facts"])
    assert any(fact["id"] == "loss_concentration" for fact in packet["facts"])
    changed_source = {**packet, "prediction_source": {**packet["prediction_source"], "model_version": "another-candidate"}}
    assert request_fingerprint(packet, "model", "") != request_fingerprint(changed_source, "model", "")
