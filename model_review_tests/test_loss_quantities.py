"""Quantity-priority regressions, including unequal production and survey scope."""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")
import harvestguard_model_v3 as hg


def profiles():
    return pd.DataFrame({
        "season": ["A", "A", "A", "A", "B", "A"],
        "crop_name": ["Maize"] * 5 + ["Tomato"],
        "district": ["Nyarugenge", "Nyagatare", "Mixed", "Mixed", "Nyagatare", "Nyagatare"],
        "production_kg": [1000., 100000., 100., 900., 200., 300.],
        hg.TARGET: [20., 3., 40., 0., 10., 20.],
    })


def test_convert_each_profile_before_summing_and_rank_quantity():
    frame = profiles()
    rates = np.array([30., 4., 50., 0., 1., 2.])
    result = hg.aggregate_losses(frame, rates)
    maize_a = result.loc[result.season.eq("A") & result.crop_name.eq("Maize")].set_index("district")
    assert maize_a.loc["Nyagatare", "predicted_loss_kg"] == 4000
    assert maize_a.loc["Nyagatare", "priority_rank"] == 1
    assert maize_a.loc["Nyarugenge", "predicted_loss_kg"] == 300
    assert maize_a.loc["Nyarugenge", "priority_rank"] == 2
    mixed = maize_a.loc["Mixed"]
    assert mixed.predicted_loss_kg == 50
    assert mixed.predicted_loss_pct == 5  # Not mean(50%, 0%) = 25%.
    assert mixed.mean_profile_predicted_loss_pct == 25
    assert mixed.observed_loss_kg == 40
    assert mixed.observed_loss_pct == 4
    assert mixed.n_profiles == 2
    assert len(result) == 5  # Seasons and crops are not combined.
    assert result.predicted_loss_kg.sum() == pytest.approx(np.dot(rates / 100, frame.production_kg))
    assert set(result.loss_quantity_scope) == {"sample_total"}
    np.testing.assert_allclose(result.predicted_loss_tonnes, result.predicted_loss_kg / 1000)


def test_survey_weights_expand_loss_and_production_once():
    frame = profiles().iloc[2:4].copy()
    frame["survey_weight"] = [10., 2.]
    result = hg.aggregate_losses(frame, [50., 0.]).iloc[0]
    assert result.predicted_loss_kg == 500
    assert result.production_kg == 2800
    assert result.predicted_loss_pct == pytest.approx(100 * 500 / 2800)
    assert result.observed_loss_kg == 400
    assert result.n_profiles == 2  # Expansion weights do not inflate support counts.
    assert result.loss_quantity_scope == "survey_expanded"


def test_unlabeled_inference_and_row_order_do_not_change_totals():
    frame = profiles().drop(columns=hg.TARGET)
    rates = np.array([30., 4., 50., 0., 1., 2.])
    expected = hg.aggregate_losses(frame, rates)
    order = [4, 2, 0, 5, 3, 1]
    shuffled = hg.aggregate_losses(frame.iloc[order], rates[order])
    pd.testing.assert_frame_equal(expected, shuffled)
    assert "observed_loss_kg" not in expected
    individual = hg.profile_loss_quantities(frame, rates)
    assert individual.predicted_loss_kg.tolist() == [300., 4000., 50., 0., 2., 6.]


def test_verified_mapping_converts_tonnes_and_keeps_production_out_of_features():
    raw = profiles().rename(columns={"production_kg": "harvest_tonnes"})
    raw.harvest_tonnes /= 1000
    raw["expansion"] = 3.
    mapped = hg.prepare_production(raw, "harvest_tonnes", "tonnes", "expansion")
    np.testing.assert_allclose(mapped.production_kg, profiles().production_kg)
    assert mapped.survey_weight.eq(3).all()
    assert not {"production_kg", "survey_weight"}.intersection(hg.feature_columns())
    with pytest.raises(ValueError, match="verified production"):
        hg.prepare_production(raw, "absent")
    with pytest.raises(ValueError, match="unit"):
        hg.prepare_production(raw, "harvest_tonnes", "bags")
    with pytest.raises(ValueError, match="Explicitly configure"):
        hg.prepare_production(mapped)


@pytest.mark.parametrize("column,value", [
    ("production_kg", None), ("production_kg", np.inf), ("production_kg", -1),
    ("production_kg", 0), ("survey_weight", None), ("survey_weight", 0),
    ("survey_weight", -2), ("survey_weight", np.inf),
])
def test_incomplete_or_invalid_production_and_weights_stop_ranking(column, value):
    frame = profiles()
    frame["survey_weight"] = 1.
    frame.loc[1, column] = value
    with pytest.raises(ValueError):
        hg.aggregate_losses(frame, [30., 4., 50., 0., 1., 2.])


@pytest.mark.parametrize("rates", [[1], [30, 101, 50, 0, 1, 2], [30, np.nan, 50, 0, 1, 2], [[1]] * 6])
def test_invalid_rate_vectors_cannot_produce_quantities(rates):
    with pytest.raises(ValueError, match="Predictions"):
        hg.aggregate_losses(profiles(), rates)


def test_ranking_evaluation_uses_kg_even_when_rate_order_is_reversed():
    frame = pd.DataFrame({"season": "A", "crop_name": "Maize", "district": list("abcdefgh"),
                          "production_kg": 1000 * 10. ** np.arange(8), hg.TARGET: np.arange(8, 0, -1.)})
    # Predictions reverse the rate ordering, but agree on the ordering of lost kg.
    result = hg.metrics(frame, np.arange(1., 9), min_profiles=1)
    assert result["ndcg5_macro"] == pytest.approx(1.)
    assert result["top5_overlap_macro"] == 1
    assert "district_mae_kg" in result
    ranks = hg.ranking_table(hg.aggregate_rates(frame, np.arange(1., 9)), min_profiles=1)
    assert ranks.iloc[0].ranking_basis == "predicted_loss_kg"


def test_equal_quantities_share_priority_rank():
    frame = profiles().iloc[:2]
    result = hg.aggregate_losses(frame, [30., .3])
    assert result.predicted_loss_kg.eq(300).all()
    assert result.priority_rank.eq(1).all()


def test_years_cannot_be_silently_combined_or_weights_used_without_production():
    frame = profiles()
    frame["year"] = [2024, 2025, 2024, 2024, 2024, 2024]
    with pytest.raises(ValueError, match="one year"):
        hg.aggregate_losses(frame, [30., 4., 50., 0., 1., 2.])
    frame = frame.drop(columns=["year", "production_kg"])
    frame["survey_weight"] = 1.
    with pytest.raises(ValueError, match="cannot substitute"):
        hg.aggregate_rates(frame, [30., 4., 50., 0., 1., 2.])


def test_notebook_embeds_the_current_pipeline_and_quantity_inputs():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "HarvestGuard_Model_Development_V3_Colab.ipynb").read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][5]["source"])
    assignment = next(node for node in ast.parse(source).body if isinstance(node, ast.Assign) and node.targets[0].id == "MODULE_SOURCE")
    assert ast.literal_eval(assignment.value) == (root / "harvestguard_model_v3.py").read_text(encoding="utf-8")
    config = "".join(notebook["cells"][4]["source"])
    assert "PRODUCTION_COLUMN" in config and "SURVEY_WEIGHT_COLUMN" in config
    assert "hg.prepare_production" in "".join(notebook["cells"][7]["source"])
    assert "hg.aggregate_losses(incoming, expected_rate)" in "".join(notebook["cells"][17]["source"])


def reuse_input():
    raw = profiles()
    raw["year"] = 2025
    raw["farmer_type"] = "smallholder"
    raw["harvest_month"] = 6
    for feature in hg.BEFORE:
        raw[feature] = 1.
    # A historical row need not have production when only 2025 is rescored.
    historical = raw.iloc[:1].copy()
    historical["year"] = 2024
    historical["production_kg"] = np.nan
    return pd.concat([raw, historical], ignore_index=True)


class ExistingModel:
    def __init__(self):
        self.predict_calls = 0

    def fit(self, *args, **kwargs):
        raise AssertionError("The existing model must never be retrained.")

    def predict(self, frame):
        self.predict_calls += 1
        assert list(frame.columns) == hg.feature_columns()
        assert not {hg.TARGET, "production_kg", "survey_weight", "year"}.intersection(frame)
        return np.where(frame.district.eq("Nyagatare"), 4., 30.)


def existing_metadata():
    return {"target": hg.TARGET, "features": hg.feature_columns(), "eligible_crops": ["Maize"],
            "feature_policy": "at_harvest", "candidate_version": "original-saved-version"}


def test_saved_model_reuse_has_no_training_or_label_leakage():
    raw = reuse_input()
    model = ExistingModel()
    profiles_out, risk, manifest = hg.quantity_results_from_saved_model(raw, model, existing_metadata(), "production_kg")
    assert model.predict_calls == 1
    assert len(profiles_out) == 5
    assert manifest["profiles_excluded_unsupported_crop"] == 1
    assert manifest["model_retrained"] is False
    assert manifest["source_candidate_version"] == "original-saved-version"
    assert set(profiles_out.year) == {2025}
    assert risk.loc[risk.season.eq("A")].iloc[0].district == "Nyagatare"
    raw[hg.TARGET] = 99
    _, changed, _ = hg.quantity_results_from_saved_model(raw, model, existing_metadata(), "production_kg")
    pd.testing.assert_series_equal(risk.predicted_loss_kg, changed.predicted_loss_kg)


def test_saved_model_reuse_needs_no_loss_labels_and_honors_unit_and_weights():
    raw = reuse_input().drop(columns=hg.TARGET)
    raw["original_tonnes"] = raw.production_kg / 1000
    raw["expansion"] = 2.
    _, risk, manifest = hg.quantity_results_from_saved_model(raw, ExistingModel(), existing_metadata(), "original_tonnes", "tonnes", "expansion")
    assert "observed_loss_kg" not in risk
    assert manifest["observed_losses_available"] is False
    assert set(risk.loss_quantity_scope) == {"survey_expanded"}
    assert risk.loc[risk.season.eq("A") & risk.district.eq("Nyagatare"), "predicted_loss_kg"].item() == 8000


def test_missing_production_error_names_the_requested_and_available_columns():
    raw = reuse_input().rename(columns={"production_kg": "original_quantity"})
    model = ExistingModel()
    with pytest.raises(ValueError) as caught:
        hg.quantity_results_from_saved_model(raw, model, existing_metadata(), "production_kg")
    message = str(caught.value)
    assert "'production_kg' was not found" in message
    assert "'original_quantity'" in message
    assert "PRODUCTION_COLUMN" in message and "SOURCE_QUANTITIES" in message
    assert model.predict_calls == 0


def test_aggregation_notebook_reuses_the_saved_model_and_embeds_current_tools():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "HarvestGuard_Quantity_Aggregation_Colab.ipynb").read_text(encoding="utf-8"))
    executable = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            tree = ast.parse("".join(cell["source"]))
            executable.append(tree)
    assignment = next(node for tree in executable for node in tree.body if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "MODULE_SOURCE")
    assert ast.literal_eval(assignment.value) == (root / "harvestguard_model_v3.py").read_text(encoding="utf-8")
    calls = [node for tree in executable for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not any(isinstance(node.func, ast.Attribute) and node.func.attr in {"fit", "search_models", "fit_locked", "prepare_data", "crop_evidence"} for node in calls)
    assert any(isinstance(node.func, ast.Attribute) and node.func.attr == "quantity_results_from_saved_model" for node in calls)


def test_aggregation_notebook_is_configured_for_the_supplied_production_file():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "HarvestGuard_Quantity_Aggregation_Colab.ipynb").read_text(encoding="utf-8"))
    config = "".join(notebook["cells"][2]["source"])
    assert "postharvest_regression_model_ready_2024_2025_with_production.csv" in config
    assert "PRODUCTION_COLUMN = 'total_crop_production'" in config
    assert "PRODUCTION_UNIT = 'kg'" in config
    conversion_checks = "".join(notebook["cells"][6]["source"])
    assert "profile_predictions.expected_loss_pct / 100 * profile_predictions.production_kg" in conversion_checks
    assert "profile_predictions.groupby(qt.KEYS)" in conversion_checks


def test_original_saved_bundle_loader_needs_no_training(tmp_path):
    import joblib
    path = tmp_path / "existing.joblib"
    joblib.dump(ExistingModel(), path)
    restored = hg.load_saved_rate_model(path)
    assert restored._quantity_loading_method == "original_joblib"
    assert restored.predict_calls == 0


def test_windows_snapshot_recovery_preserves_exact_inference(tmp_path, monkeypatch):
    import joblib
    from types import SimpleNamespace
    x = np.arange(12., dtype=np.float32).reshape(-1, 1)
    y = np.array([0.] * 6 + [1.] * 6)
    # Small synthetic fit only constructs a test artifact; the loader must not fit.
    original = hg.XGBRegressor(n_estimators=2, max_depth=1, n_jobs=1).fit(x, y).get_booster()
    expected = original.predict(hg.xgboost.DMatrix(x))
    path = tmp_path / "trusted_snapshot.joblib"
    joblib.dump(original, path)
    real_load = joblib.load
    original_setstate = hg.xgboost.Booster.__setstate__
    calls = []

    def simulate_windows_rng_error(path):
        calls.append(path)
        if len(calls) == 1:
            raise hg.xgboost.core.XGBoostError("input stream corrupted")
        return real_load(path)

    monkeypatch.setattr(hg, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(hg.joblib, "load", simulate_windows_rng_error)
    restored = hg.load_saved_rate_model(path)
    np.testing.assert_array_equal(expected, restored.predict(hg.xgboost.DMatrix(x)))
    assert original.get_dump(dump_format="json") == restored.get_dump(dump_format="json")
    assert restored._quantity_inference_only is True
    assert hg.xgboost.Booster.__setstate__ is original_setstate


def test_bad_snapshot_is_rejected_and_the_original_loader_is_restored(monkeypatch):
    from types import SimpleNamespace
    original_setstate = hg.xgboost.Booster.__setstate__
    calls = []

    def bad_snapshot(path):
        calls.append(path)
        if len(calls) == 1:
            raise hg.xgboost.core.XGBoostError("input stream corrupted")
        hg.xgboost.Booster.__setstate__(object(), {"handle": b"not-a-model"})

    monkeypatch.setattr(hg, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(hg.joblib, "load", bad_snapshot)
    with pytest.raises(ValueError, match="supported Config/Model"):
        hg.load_saved_rate_model("unused")
    assert hg.xgboost.Booster.__setstate__ is original_setstate
