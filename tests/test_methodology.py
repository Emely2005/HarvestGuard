"""Public presentation must preserve the recorded model's actual results."""
import json
from pathlib import Path

import pandas as pd
import pytest

from harvestguard_methodology import SOURCES, matched_importance, validation_summary

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "model_v3_candidate"


def test_validation_reports_only_selected_model_and_exact_recorded_errors():
    summary = validation_summary(MODEL)
    assert summary.shape == (2, 3)
    assert summary.iloc[0]["Profile error (percentage points)"] == pytest.approx(0.4391498217242469)
    assert summary.iloc[1]["Profile error (percentage points)"] == pytest.approx(0.48625333301566415)
    assert "baseline" not in summary.to_string().lower()
    assert not any("NDCG" in column for column in summary.columns)


def test_survey_references_and_supplied_documentation_are_available():
    assert [source[1] for source in SOURCES] == ["RWA-NISR-SAS-2024-v01", "RWA-NISR-SAS-2025-v01"]
    for title, reference, url, filename in SOURCES:
        assert url.startswith("https://microdata.statistics.gov.rw/")
        assert (ROOT / filename).read_bytes().startswith(b"%PDF")


def test_importance_is_original_model_matched_and_normalised():
    importance = matched_importance(MODEL)
    assert importance.combined_importance.sum() == pytest.approx(1)
    assert importance.iloc[0]["label"] == "Crop"
    assert "temperature_mean_14d_after_c" not in importance.feature.tolist()
    assert "humidity_mean_7d_before_pct" in importance.feature.tolist()


def test_preprocessing_and_displayed_feature_policies_are_distinguished():
    prepared = json.loads((ROOT / "source_preprocessing/postharvest_regression_model_manifest.json").read_text())
    deployed = json.loads((MODEL / "model_v3_metadata.json").read_text())
    assert "temperature_mean_14d_after_c" in prepared["features_X"]
    assert "temperature_mean_14d_after_c" not in deployed["features"]
    assert len(deployed["features"]) == 10
