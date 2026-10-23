"""Connect verified real quantity results to the app without changing the model."""
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from harvestguard_data import (
    MODEL_FOLDER, QUANTITY_FOLDER, QUANTITY_METADATA_FILE, QUANTITY_PREDICTIONS_FILE,
    REQUIRED_FILES, dashboard_signature, load_dashboard_data, load_model_data, rank_districts,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def dashboard_bundle(tmp_path):
    model = tmp_path / MODEL_FOLDER
    model.mkdir()
    for filename in REQUIRED_FILES:
        shutil.copyfile(ROOT / MODEL_FOLDER / filename, model / filename)
    quantities = tmp_path / QUANTITY_FOLDER
    quantities.mkdir()
    for filename in (QUANTITY_PREDICTIONS_FILE, QUANTITY_METADATA_FILE):
        shutil.copyfile(ROOT / QUANTITY_FOLDER / filename, quantities / filename)
    return model, quantities


def test_app_loads_real_kg_priorities_and_retains_original_validation(dashboard_bundle):
    model, quantities = dashboard_bundle
    risk, metadata, readiness = load_dashboard_data(model)
    expected = pd.read_csv(quantities / QUANTITY_PREDICTIONS_FILE)
    pd.testing.assert_frame_equal(risk, expected)
    assert metadata["ranking_basis"] == "predicted_loss_kg"
    assert metadata["validation_ranking_basis"] == "predicted_loss_pct"
    assert metadata["quantity_ranking_validation_recomputed"] is False
    assert metadata["model_ready_for_national_allocation"] is False
    assert metadata["prediction_file"].endswith(QUANTITY_PREDICTIONS_FILE)
    assert readiness.passed.tolist() == [True, False, False]
    view = rank_districts(risk.loc[risk.season.eq("A") & risk.crop_name.eq("Maize")])
    assert view.iloc[0].district == "Nyagatare"
    assert view.iloc[0].predicted_loss_kg == pytest.approx(64545.966802, abs=1e-5)
    nyarugenge = view.loc[view.district.eq("Nyarugenge")].iloc[0]
    assert nyarugenge.predicted_loss_pct > view.iloc[0].predicted_loss_pct
    assert nyarugenge.predicted_loss_kg < 50
    assert nyarugenge.display_rank == 26
    original, _, _ = load_model_data(model)
    assert "predicted_loss_kg" not in original
    assert original.n_profiles.sum() == risk.n_profiles.sum() == 50475


@pytest.mark.parametrize("problem", ["model_hash", "candidate", "year", "features", "ranking_basis", "profile_count", "missing_metadata"])
def test_quantity_source_must_match_the_saved_model(dashboard_bundle, problem):
    model, quantities = dashboard_bundle
    path = quantities / QUANTITY_METADATA_FILE
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if problem == "missing_metadata":
        path.unlink()
    else:
        if problem == "model_hash":
            metadata["source_model_sha256"] = "other-artifact"
        elif problem == "candidate":
            metadata["source_candidate_version"] = "another-candidate"
        elif problem == "year":
            metadata["year"] = 2024
        elif problem == "features":
            metadata["features"] = []
        elif problem == "ranking_basis":
            metadata["ranking_basis"] = "predicted_loss_pct"
        elif problem == "profile_count":
            metadata["profiles_scored"] = 1
        path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError)):
        load_dashboard_data(model)


def test_downloaded_colab_files_work_in_the_project_root(dashboard_bundle):
    model, quantities = dashboard_bundle
    for filename in (QUANTITY_PREDICTIONS_FILE, QUANTITY_METADATA_FILE):
        (quantities / filename).replace(model.parent / filename)
    risk, metadata, _ = load_dashboard_data(model)
    assert len(risk) == 944
    assert metadata["prediction_file"] == QUANTITY_PREDICTIONS_FILE


def test_quantity_updates_invalidate_the_dashboard_cache(dashboard_bundle):
    model, quantities = dashboard_bundle
    before = dashboard_signature(model)
    path = quantities / QUANTITY_METADATA_FILE
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert dashboard_signature(model) != before


def test_no_quantity_results_retains_a_labelled_legacy_view(dashboard_bundle):
    model, quantities = dashboard_bundle
    for filename in (QUANTITY_PREDICTIONS_FILE, QUANTITY_METADATA_FILE):
        (quantities / filename).unlink()
    risk, metadata, _ = load_dashboard_data(model)
    assert "predicted_loss_kg" not in risk
    assert metadata["ranking_basis"] == "predicted_loss_pct"
