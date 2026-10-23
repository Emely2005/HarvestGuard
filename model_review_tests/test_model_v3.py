from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")
pytest.importorskip("optuna")
import harvestguard_model_v3 as hg


def synthetic_data(seed=12):
    rng = np.random.default_rng(seed)
    rows = []
    for year in [2024, 2025]:
        for season in ["A", "B", "C"]:
            for crop in ["Maize", "Tomato"]:
                for district in range(8):
                    for profile in range(15):
                        row = dict(year=year, season=season, district=f"District {district}", crop_name=crop,
                                   farmer_type="smallholder", harvest_month=(district % 6) + 1,
                                   n_plots_crop=int(rng.integers(1, 5)), crop_plot_area_ha_total=float(rng.uniform(.1, 3)),
                                   num_crops_in_plot=1, rain_30d_before_mm=float(rng.uniform(0, 120)),
                                   humidity_mean_7d_before_pct=float(rng.uniform(30, 95)),
                                   household_id=f"{district}-{profile}")
                        row.update({name: float(rng.uniform(1, 30)) for name in hg.AFTER})
                        probability = .15 + .03 * district + (.12 if crop == "Tomato" else 0)
                        row[hg.TARGET] = float(rng.gamma(1.5, 1 + district / 4)) if rng.random() < probability else 0.
                        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def data():
    return hg.prepare_data(synthetic_data())


def test_target_and_prediction_time_features_are_explicit():
    raw = synthetic_data()
    raw.loc[0, hg.TARGET] = np.nan
    with pytest.raises(ValueError, match="do not fill"):
        hg.prepare_data(raw)
    assert not set(hg.AFTER).intersection(hg.feature_columns("at_harvest"))
    assert set(hg.AFTER).issubset(hg.feature_columns("retrospective"))
    assert hg.TARGET not in hg.feature_columns()
    assert "year" not in hg.feature_columns()


def test_missing_covariates_are_allowed_but_target_is_not(data):
    data.loc[0, "rain_30d_before_mm"] = np.nan
    dev = data.loc[data.year.eq(2024)].reset_index(drop=True)
    part = hg.prepare_partition(dev, None, hg.feature_columns())
    assert np.isfinite(part["base_x"].toarray() if hasattr(part["base_x"], "toarray") else part["base_x"]).all()


def test_splits_keep_independent_groups_and_2025_out(data):
    dev = data.loc[data.year.eq(2024)].reset_index(drop=True)
    for entity in [None, "household_id"]:
        group = hg.groups(dev, entity)
        counts = np.zeros(len(dev), int)
        for train, valid in hg.cv_splits(dev, entity_column=entity):
            assert set(group.iloc[train]).isdisjoint(group.iloc[valid])
            counts[valid] += 1
        assert (counts == 1).all()
    with pytest.raises(ValueError, match="2025"):
        hg.cv_splits(data)


def test_history_does_not_read_validation_labels_and_shrinkage_is_bounded(data):
    dev, test = data.loc[data.year.eq(2024)], data.loc[data.year.eq(2025)].copy()
    for alpha in [0, 50, 200]:
        fitted = hg.HistoricalRate(alpha).fit(dev)
        rate, probability, severity = fitted.predict_components(test)
        test[hg.TARGET] = 99
        np.testing.assert_allclose(rate, fitted.predict(test))
        np.testing.assert_allclose(rate, probability * severity)
        assert np.isfinite(rate).all() and ((rate >= 0) & (rate <= 100)).all()


def test_tied_top_five_is_fair_and_constant_observed_is_not_perfect():
    agg = pd.DataFrame({"season": "A", "crop_name": "Maize", "district": [str(i) for i in range(10)],
                        "observed": np.arange(10), "predicted": 1., "profiles": 20})
    first = hg.ranking_table(agg).iloc[0]
    shuffled = hg.ranking_table(agg.sample(frac=1, random_state=5)).iloc[0]
    assert first.top5_overlap == pytest.approx(.5)
    assert first.ndcg5 == shuffled.ndcg5
    assert first.spearman == 0
    agg["observed"] = 0
    result = hg.ranking_table(agg)
    assert result.excluded_reason.iloc[0] == "constant_observed_rates"


@pytest.mark.parametrize("family,severity", [("hurdle", "gamma"), ("hurdle", "tweedie"), ("tweedie", "gamma")])
def test_held_out_labels_never_choose_trees_or_calibration(data, family, severity):
    dev = data.loc[data.year.eq(2024)].reset_index(drop=True)
    train, valid = hg.cv_splits(dev)[0]
    outer = dev.iloc[valid].copy()
    params = hg.default_params()
    params.update(family=family, severity_objective=severity)
    part1 = hg.prepare_partition(dev.iloc[train], outer, hg.feature_columns())
    model1 = hg.fit_rate_model(part1, hg.feature_columns(), params, max_trees=6, threads=1)
    outer[hg.TARGET] = 99
    part2 = hg.prepare_partition(dev.iloc[train], outer, hg.feature_columns())
    model2 = hg.fit_rate_model(part2, hg.feature_columns(), params, max_trees=6, threads=1)
    np.testing.assert_allclose(model1.predict(outer), model2.predict(outer), rtol=1e-7)
    assert model1.diagnostics == model2.diagnostics
    rates, probability, severity = model1.predict_components(outer)
    if family == "hurdle":
        np.testing.assert_allclose(rates, probability * severity)
    assert ((rates >= 0) & (rates <= 100)).all()


def test_quantity_target_check_requires_explicit_verified_columns():
    raw = pd.DataFrame({"prod": [1000], "store": [2], "process": [1], "pack": [1], hg.TARGET: [.4]})
    mapping = {"production": "prod", "storage": "store", "processing": "process", "packaging": "pack"}
    assert not hg.audit_target_quantities(raw, mapping).mismatch.any()
    raw.loc[0, "prod"] = 0
    with pytest.raises(ValueError, match="Zero-production"):
        hg.audit_target_quantities(raw, mapping)


@pytest.mark.parametrize("quantity", [False, True])
def test_search_lock_serialization_and_temporal_export(data, tmp_path, quantity):
    if quantity:
        data = data.copy()
        data["production_kg"] = 100 + 500 * (np.arange(len(data)) % 13)
        data["survey_weight"] = 1 + np.arange(len(data)) % 3
    dev = data.loc[data.year.eq(2024)].reset_index(drop=True)
    test = data.loc[data.year.eq(2025)].reset_index(drop=True)
    lock, oof, table = hg.search_models(dev, hg.feature_columns(), tmp_path, n_trials=2, max_trees=5, n_splits=2, min_profiles=5, threads=1)
    assert len(oof) == len(dev) and np.isfinite(oof).all()
    assert "Always zero (diagnostic only)" not in lock["selection"]["model"]
    assert (tmp_path / "locked_choice_before_2025.json").exists()
    fitted = hg.fit_locked(dev, lock, threads=1)
    with pytest.raises(ValueError, match="2024"):
        hg.fit_locked(test, lock)
    path = tmp_path / "bundle.joblib"
    joblib.dump(fitted, path)
    np.testing.assert_allclose(fitted.predict(test), joblib.load(path).predict(test))
    unlabeled = test.drop(columns=[hg.TARGET, "year"])
    prepared = hg.prepare_features(unlabeled, lock["features"])
    np.testing.assert_allclose(fitted.predict(test), joblib.load(path).predict(prepared))
    comparisons, risk, calibration, checks = hg.evaluate_locked(dev, test, fitted, lock, tmp_path)
    assert len(risk) == 48
    assert {"district", "crop_name", "season", "predicted_loss_pct", "n_profiles", "support_level"}.issubset(risk.columns)
    assert comparisons.ranking_cohorts.gt(0).all()
    if quantity:
        assert lock["config"]["ranking_basis"] == "predicted_loss_kg"
        assert "district_mae_kg" in comparisons
        assert set(risk.loss_quantity_scope) == {"survey_expanded"}
        expected = hg.profile_loss_quantities(test, fitted.predict(test))
        assert risk.predicted_loss_kg.sum() == pytest.approx(expected.weighted_predicted_loss_kg.sum())
        pd.testing.assert_series_equal(risk.priority_rank, risk.vulnerability_rank, check_names=False)
    assert len(checks) == 3
    assert (tmp_path / "component_diagnostics_2025.csv").exists()
    # Same configuration resumes instead of silently repeating the budget.
    second, _, _ = hg.search_models(dev, hg.feature_columns(), tmp_path, n_trials=2, max_trees=5, n_splits=2, min_profiles=5, threads=1)
    assert second["choice"] == lock["choice"]
