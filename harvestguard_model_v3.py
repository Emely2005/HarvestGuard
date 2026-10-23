"""Colab candidate pipeline: expected loss rates, honest baselines and ranking.

This is a research candidate, separate from the running Streamlit application.
The model predicts percentages (0 to 100). Intervention ranking sums profile
loss quantities after multiplying each prediction by its production denominator.
"""

from pathlib import Path
import hashlib
import json
import os
import platform
import time

import joblib
import numpy as np
import pandas as pd
import scipy
from scipy.special import expit, logit
from scipy.stats import spearmanr
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, mean_tweedie_deviance, ndcg_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
import xgboost
from xgboost import XGBClassifier, XGBRegressor


TARGET = "actionable_loss_rate_pct"
KEYS = ["season", "crop_name", "district"]
CATS = ["season", "district", "crop_name", "farmer_type", "harvest_month"]
BEFORE = ["n_plots_crop", "crop_plot_area_ha_total", "num_crops_in_plot", "rain_30d_before_mm", "humidity_mean_7d_before_pct"]
AFTER = ["rain_14d_after_mm", "rainy_days_14d_after", "humidity_mean_14d_after_pct", "temperature_mean_14d_after_c"]
SEED = 42
CODE_VERSION = "harvestguard-v3-candidate-2"


def versions():
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__, "xgboost": xgboost.__version__}


def load_saved_rate_model(path):
    """Load a trusted saved bundle for inference, including Linux-to-Windows IO.

    XGBoost issue #12459 documents a nonportable RNG state in pickled boosters.
    For that exact Windows error only, recover the unchanged Model section via
    XGBoost's model loader. Training configuration/RNG state is not required for
    inference. No trees, objectives, calibration or preprocessing are refitted.
    https://github.com/dmlc/xgboost/issues/12459
    """
    try:
        fitted = joblib.load(path)
        fitted._quantity_loading_method = "original_joblib"
        return fitted
    except xgboost.core.XGBoostError as error:
        if os.name != "nt" or "input stream corrupted" not in str(error):
            raise
    original = xgboost.Booster.__setstate__
    recovered = []

    def restore_model_only(booster, state):
        snapshot = bytes(state["handle"])
        # XGBoost's UBJSON snapshot has Config then Model at the top level.
        # Pass the original Model bytes directly to the native model loader:
        # no tree values are decoded, edited or reserialized by Python.
        marker = b"L" + (5).to_bytes(8, "big", signed=True) + b"Model"
        learner = b"{L" + (7).to_bytes(8, "big", signed=True) + b"learner"
        if snapshot[:1] != b"{" or snapshot[-1:] != b"}" or snapshot.count(marker) != 1:
            raise ValueError("The saved XGBoost snapshot is not the supported Config/Model UBJSON layout. Use the original Colab environment to export portable model files.")
        start = snapshot.index(marker) + len(marker)
        model_buffer = snapshot[start:-1]  # Remove only the snapshot's outer closing brace.
        if not model_buffer.startswith(learner) or model_buffer[-1:] != b"}":
            raise ValueError("The saved snapshot has no complete final Model section. Use the original Colab environment to export portable model files.")
        xgboost.Booster.__init__(booster, model_file=bytearray(model_buffer))
        booster.__dict__.update({key: value for key, value in state.items() if key != "handle"})
        recovered.append(1)

    # The replacement is scoped to loading this bundle and is always restored.
    try:
        xgboost.Booster.__setstate__ = restore_model_only
        fitted = joblib.load(path)
    finally:
        xgboost.Booster.__setstate__ = original
    if not recovered:
        raise ValueError("No saved boosters were recovered; the original loading error remains unresolved.")
    fitted._quantity_loading_method = "existing_model_sections_from_windows_incompatible_rng_snapshot"
    fitted._quantity_inference_only = True
    return fitted


def feature_columns(mode="at_harvest"):
    if mode not in {"at_harvest", "retrospective"}:
        raise ValueError("Choose at_harvest or retrospective; never silently mix feature policies.")
    return CATS + BEFORE + (AFTER if mode == "retrospective" else [])


def prepare_features(raw, features):
    """The same covariate normalization for training and unlabeled inference."""
    missing = set(features).difference(raw.columns)
    if missing:
        raise ValueError("Required columns missing: " + ", ".join(sorted(missing)))
    frame = raw.copy()
    for name in CATS:
        if name == "harvest_month":
            month = pd.to_numeric(frame[name], errors="coerce")
            month = month.where(month.isin(range(1, 13)))
            frame[name] = month.astype("Int64").astype("string").fillna("_MISSING_").astype(str)
        else:
            frame[name] = frame[name].astype("string").str.strip().fillna("_MISSING_").astype(str)
    for name in set(features).difference(CATS):
        frame[name] = pd.to_numeric(frame[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if frame[KEYS].eq("_MISSING_").any().any() or frame[KEYS].eq("").any().any():
        raise ValueError("District, crop and season identities cannot be missing.")
    return frame.reset_index(drop=True)


def prepare_data(raw, mode="at_harvest"):
    """Validate rates without imputing missing outcomes or discarding true zeros."""
    if TARGET not in raw or "year" not in raw:
        raise ValueError("Development data requires observed targets and year.")
    frame = prepare_features(raw, feature_columns(mode))
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
    if not np.isfinite(frame[TARGET].to_numpy()).all() or not frame[TARGET].between(0, 100).all():
        raise ValueError("Targets must be observed finite percentages in [0,100]. Audit missing outcomes and units; do not fill them with zero or clip labels.")
    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
    if not {2024, 2025}.issubset(set(frame.year)):
        raise ValueError("Both 2024 development and 2025 temporal-validation rows are required.")
    return frame


def audit_target_quantities(raw, columns, tolerance_pp=1e-5):
    """Optional check only when original quantities share a verified unit/scope."""
    required = {"production", "storage", "processing", "packaging"}
    if set(columns) != required:
        raise ValueError("Map production, storage, processing and packaging to their source column names.")
    quantities = raw[list(columns.values())].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(quantities.to_numpy()).all() or (quantities < 0).any().any():
        raise ValueError("Source quantities contain missing, non-finite or negative values; resolve their meaning first.")
    production = quantities[columns["production"]]
    loss = sum(quantities[columns[k]] for k in ["storage", "processing", "packaging"])
    if (production <= 0).any():
        raise ValueError("Zero-production records require an explicit target policy; do not silently divide by zero.")
    computed = 100 * loss / production
    existing = pd.to_numeric(raw[TARGET], errors="coerce")
    bad = ~np.isclose(computed, existing, atol=tolerance_pp, rtol=1e-6)
    return pd.DataFrame({"recomputed_loss_pct": computed, "saved_loss_pct": existing, "mismatch": bad})


def prepare_production(raw, production_column="production_kg", unit="kg", survey_weight_column=None):
    """Map verified profile/crop/season production to kg; never infer it from area.

    Production must be the same denominator as TARGET, once per profile. Survey
    weights, when supplied, must be verified expansion weights, not model weights.
    Neither production nor the weights are added to the predictive feature list.
    """
    factors = {"kg": 1., "tonnes": 1000.}
    if unit not in factors:
        raise ValueError("Production unit must be 'kg' or 'tonnes' (metric tonnes).")
    if not production_column or production_column not in raw:
        available = ", ".join(repr(str(name)) for name in raw.columns)
        raise ValueError(
            f"Configured production column {production_column!r} was not found. "
            f"Available CSV columns: {available}. "
            "Quantity ranking requires a verified production column for every profile. "
            "Set PRODUCTION_COLUMN to its exact CSV name and PRODUCTION_UNIT to 'kg' or 'tonnes'. "
            "If production was removed during cleaning, restore the target's original production denominator "
            "using verified profile/crop/season/year identifiers. Renaming an unrelated column, "
            "using area or profile count, or changing SOURCE_QUANTITIES will not recover production."
        )
    frame = raw.copy()
    production = pd.to_numeric(frame[production_column], errors="coerce").to_numpy(float) * factors[unit]
    if not np.isfinite(production).all() or (production <= 0).any():
        raise ValueError("Production must be finite, positive and observed on every profile; do not impute missing production or silently drop rows.")
    frame["production_kg"] = production
    if survey_weight_column is not None:
        if survey_weight_column not in raw:
            raise ValueError("The configured survey expansion weight column is missing.")
        weights = pd.to_numeric(raw[survey_weight_column], errors="coerce").to_numpy(float)
        if not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("Survey expansion weights must be observed, finite and positive on every profile.")
        frame["survey_weight"] = weights
    elif "survey_weight" in frame:
        raise ValueError("Explicitly configure survey_weight_column to use existing survey_weight values.")
    return frame


def profile_loss_quantities(frame, predicted):
    """Convert each rate before aggregation; also works without observed labels."""
    if frame.empty:
        raise ValueError("Quantity ranking requires at least one profile.")
    if "year" in frame and (frame.year.isna().any() or frame.year.nunique() != 1):
        raise ValueError("Aggregate one year at a time; do not combine the same season across different years.")
    predicted = np.asarray(predicted, float)
    if predicted.shape != (len(frame),) or not np.isfinite(predicted).all() or ((predicted < 0) | (predicted > 100)).any():
        raise ValueError("Predictions must match rows and be finite percentages in [0,100].")
    if "production_kg" not in frame:
        raise ValueError("Quantity ranking requires production_kg on every profile.")
    production = pd.to_numeric(frame.production_kg, errors="coerce").to_numpy(float)
    weights = pd.to_numeric(frame.survey_weight, errors="coerce").to_numpy(float) if "survey_weight" in frame else np.ones(len(frame))
    if not np.isfinite(production).all() or (production <= 0).any():
        raise ValueError("Production must be finite, positive and observed on every profile.")
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Survey expansion weights must be observed, finite and positive.")
    if frame[KEYS].isna().any().any() or frame[KEYS].astype(str).apply(lambda s: s.str.strip().eq("")).any().any():
        raise ValueError("District, crop and season identities cannot be missing.")
    z = frame[KEYS].copy()
    z["production_kg"] = production
    z["survey_weight"] = weights
    z["expected_loss_pct"] = predicted
    z["predicted_loss_kg"] = (predicted / 100) * production
    z["weighted_production_kg"] = production * weights
    z["weighted_predicted_loss_kg"] = z.predicted_loss_kg * weights
    if TARGET in frame:
        observed = pd.to_numeric(frame[TARGET], errors="coerce").to_numpy(float)
        if not np.isfinite(observed).all() or ((observed < 0) | (observed > 100)).any():
            raise ValueError("Observed loss rates must be finite percentages in [0,100].")
        z["observed_loss_pct"] = observed
        z["observed_loss_kg"] = (observed / 100) * production
        z["weighted_observed_loss_kg"] = z.observed_loss_kg * weights
    numeric = z.select_dtypes(include="number").to_numpy()
    if not np.isfinite(numeric).all():
        raise ValueError("Loss quantity arithmetic overflowed; verify production units and weights.")
    return z


def aggregate_losses(frame, predicted):
    """Sum profile losses by season/crop/district, using expansion weights if given.

    Without survey weights the quantities cover the sample only. A group mean
    percentage times a district production total is not used here.
    """
    z = profile_loss_quantities(frame, predicted)
    fields = dict(production_kg=("weighted_production_kg", "sum"),
                  predicted_loss_kg=("weighted_predicted_loss_kg", "sum"),
                  n_profiles=("expected_loss_pct", "size"),
                  mean_profile_predicted_loss_pct=("expected_loss_pct", "mean"))
    if TARGET in frame:
        fields.update(observed_loss_kg=("weighted_observed_loss_kg", "sum"),
                      mean_profile_observed_loss_pct=("observed_loss_pct", "mean"))
    result = z.groupby(KEYS, as_index=False).agg(**fields)
    for kind in ["predicted"] + (["observed"] if TARGET in frame else []):
        result[f"{kind}_loss_pct"] = 100 * result[f"{kind}_loss_kg"] / result.production_kg
        result[f"{kind}_loss_tonnes"] = result[f"{kind}_loss_kg"] / 1000
    if not np.isfinite(result.select_dtypes(include="number").to_numpy()).all():
        raise ValueError("Aggregated loss quantities must be finite; verify production units and weights.")
    result["loss_quantity_scope"] = "survey_expanded" if "survey_weight" in frame else "sample_total"
    result["priority_rank"] = result.groupby(["season", "crop_name"]).predicted_loss_kg.rank(method="min", ascending=False).astype(int)
    return result


def quantity_results_from_saved_model(raw, model, metadata, production_column=None, unit="kg",
                                      survey_weight_column=None, year=2025):
    """Predict and aggregate with an existing model, without fitting or tuning.

    Use the saved eligibility and features. Loss labels are optional and are
    never passed to the model. Production is required only on the scored year's
    eligible profiles, not on historical training rows that are not rescored.
    """
    if metadata.get("target") != TARGET:
        raise ValueError("The saved model must predict actionable_loss_rate_pct.")
    features, crops = metadata.get("features"), metadata.get("eligible_crops")
    if not isinstance(features, list) or not features or not isinstance(crops, list) or not crops:
        raise ValueError("Saved metadata must contain the original features and eligible_crops.")
    if set(features) != set(feature_columns(metadata.get("feature_policy", "at_harvest"))):
        raise ValueError("Saved feature definitions differ from the supported V3 feature policy.")
    if "year" not in raw or "crop_name" not in raw:
        raise ValueError("Farmer-level input must include year and crop_name; do not use the district-average prediction CSV.")
    raw = raw.reset_index(drop=True)
    years = pd.to_numeric(raw.year, errors="coerce")
    if not np.isfinite(years.to_numpy(float)).all() or (years % 1 != 0).any():
        raise ValueError("Input year must be observed whole numbers on every row.")
    year_rows = raw.loc[years.eq(year)].copy()
    if year_rows.empty:
        raise ValueError(f"No farmer-level rows were found for {year}.")
    crop_names = year_rows.crop_name.astype("string").str.strip()
    selected = year_rows.loc[crop_names.isin(crops)].copy()
    selected["crop_name"] = crop_names.loc[selected.index]
    selected["year"] = year
    if selected.empty:
        raise ValueError("No input profiles match the saved model's eligible crops.")
    selected = prepare_production(selected, production_column, unit, survey_weight_column)
    incoming = prepare_features(selected, features)
    # Explicit feature selection prevents labels or production from becoming model inputs.
    rate = np.asarray(model.predict(incoming[features]), float)
    profiles = profile_loss_quantities(incoming, rate)
    profiles.insert(0, "year", year)
    risk = aggregate_losses(incoming, rate)
    risk["vulnerability_rank"] = risk.priority_rank
    risk["support_level"] = pd.cut(risk.n_profiles, [-1, 9, 29, 99, np.inf], labels=["Limited", "Low", "Moderate", "Strong"]).astype(str)
    risk = risk.sort_values(["season", "crop_name", "priority_rank", "district"]).reset_index(drop=True)
    manifest = {
        "operation": "existing_model_inference_and_quantity_aggregation",
        "model_retrained": False,
        "model_loading_method": getattr(model, "_quantity_loading_method", "provided_existing_model"),
        "source_candidate_version": metadata.get("candidate_version", "unspecified"),
        "champion": metadata.get("champion", "unspecified"),
        "target": TARGET, "features": features, "feature_policy": metadata.get("feature_policy"),
        "year": int(year), "profiles_in_year": len(year_rows), "profiles_scored": len(selected),
        "profiles_excluded_unsupported_crop": len(year_rows) - len(selected),
        "production_source_column": production_column, "production_source_unit": unit,
        "survey_weight_source_column": survey_weight_column,
        "ranking_basis": "predicted_loss_kg",
        "loss_quantity_scope": "survey_expanded" if survey_weight_column is not None else "sample_total",
        "observed_losses_available": TARGET in selected,
        "quantity_ranking_validation_recomputed": False,
        "model_ready_for_national_allocation": False,
        "note": "Saved model and historical rate-validation reports are unchanged. These results do not establish new ranking accuracy or intervention benefits.",
    }
    return profiles, risk, manifest


def crop_evidence(dev):
    if not dev.year.eq(2024).all():
        raise ValueError("Eligibility must be defined from 2024 only.")
    table = dev.groupby("crop_name").agg(n=(TARGET, "size"), positives=(TARGET, lambda y: int((y > 0).sum())),
                                        districts=("district", "nunique"), seasons=("season", "nunique")).reset_index()
    table["eligible"] = (table.n >= 200) & (table.positives >= 10) & (table.districts >= 5) & (table.seasons >= 2)
    table["evidence_warning"] = np.where(table.positives < 30, "Very few positive cases: avoid crop-specific fitting", "")
    return table


def groups(frame, entity_column=None):
    if entity_column:
        if entity_column not in frame or frame[entity_column].isna().any():
            raise ValueError("The configured entity ID must exist and be observed on every row.")
        # IDs are assumed unique within district; they are never predictive features.
        return frame["district"].astype(str) + "|entity|" + frame[entity_column].astype(str)
    return frame[KEYS].astype(str).agg("|".join, axis=1)


def cv_splits(dev, n_splits=3, entity_column=None):
    if not dev.year.eq(2024).all():
        raise ValueError("2025 outcomes must not enter development splits.")
    group = groups(dev, entity_column)
    if group.nunique() < n_splits:
        raise ValueError("Too few independent groups for the requested folds.")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    splits = list(splitter.split(dev, dev[TARGET].gt(0), group))
    coverage = np.zeros(len(dev), dtype=int)
    for train, valid in splits:
        assert set(group.iloc[train]).isdisjoint(group.iloc[valid])
        coverage[valid] += 1
    assert (coverage == 1).all()
    return splits


def aggregate_rates(frame, predicted):
    if "production_kg" in frame:
        return aggregate_losses(frame, predicted).rename(columns={"predicted_loss_pct": "predicted", "observed_loss_pct": "observed", "n_profiles": "profiles"})
    if "survey_weight" in frame:
        raise ValueError("Survey weights cannot substitute for missing production_kg.")
    predicted = np.asarray(predicted, float)
    if len(predicted) != len(frame) or not np.isfinite(predicted).all() or ((predicted < 0) | (predicted > 100)).any():
        raise ValueError("Predictions must match rows and be finite percentages in [0,100].")
    z = frame[KEYS + [TARGET]].copy()
    z["predicted"] = predicted
    return z.groupby(KEYS, as_index=False).agg(observed=(TARGET, "mean"), predicted=("predicted", "mean"), profiles=(TARGET, "size"))


def _top_membership(values, k):
    values = np.asarray(values, float)
    cutoff = np.sort(values)[-k]
    above, equal = values > cutoff, values == cutoff
    return above.astype(float) + equal * ((k - above.sum()) / equal.sum())


def ranking_table(agg, min_profiles=10, min_districts=5):
    quantity = "predicted_loss_kg" in agg
    observed, predicted = ("observed_loss_kg", "predicted_loss_kg") if quantity else ("observed", "predicted")
    rows = []
    for (season, crop), all_rows in agg.groupby(["season", "crop_name"]):
        part = all_rows.loc[all_rows.profiles >= min_profiles]
        reason = ""
        if len(part) < min_districts:
            reason = "too_few_supported_districts"
        elif np.ptp(part[observed].to_numpy()) < 1e-12:
            reason = "constant_observed_losses" if quantity else "constant_observed_rates"
        row = {"season": season, "crop_name": crop, "districts": len(part), "excluded_reason": reason,
               "ranking_basis": "predicted_loss_kg" if quantity else "predicted_loss_pct"}
        if not reason:
            y, p = part[observed].to_numpy(), part[predicted].to_numpy()
            k = min(5, len(part))
            row.update(ndcg5=float(ndcg_score(y[None, :], p[None, :], k=k, ignore_ties=False)),
                       spearman=0.0 if np.ptp(p) < 1e-12 else float(spearmanr(y, p).statistic),
                       top5_overlap=float(np.dot(_top_membership(y, k), _top_membership(p, k)) / k))
        rows.append(row)
    return pd.DataFrame(rows)


def metrics(frame, predicted, min_profiles=10):
    y, p = frame[TARGET].to_numpy(float), np.asarray(predicted, float)
    agg = aggregate_rates(frame, p)
    supported = agg.loc[agg.profiles >= min_profiles]
    ranks = ranking_table(agg, min_profiles)
    usable = ranks.loc[ranks.excluded_reason.eq("")]
    mean_y = float(y.mean())
    denominator = float(np.sum((y - mean_y) ** 2))
    result = {
        "mae_pp": float(np.mean(np.abs(y - p))), "rmse_pp": float(np.sqrt(np.mean((y - p) ** 2))),
        "r2": float(1 - np.sum((y - p) ** 2) / denominator) if denominator > 0 else np.nan,
        "mean_predicted_pct": float(p.mean()), "mean_observed_pct": mean_y,
        "district_mae_pp": float(np.mean(np.abs(supported.observed - supported.predicted))) if len(supported) else np.nan,
        "district_rmse_pp": float(np.sqrt(np.mean((supported.observed - supported.predicted) ** 2))) if len(supported) else np.nan,
        "tweedie_deviance_floor_0_001pp": float(mean_tweedie_deviance(y / 100, np.maximum(p, .001) / 100, power=1.5)),
        "ndcg5_macro": float(usable.ndcg5.mean()) if len(usable) else np.nan,
        "spearman_macro": float(usable.spearman.mean()) if len(usable) else np.nan,
        "top5_overlap_macro": float(usable.top5_overlap.mean()) if len(usable) else np.nan,
        "ranking_cohorts": int(len(usable)), "supported_district_groups": int(len(supported)),
    }
    if "predicted_loss_kg" in agg:
        error = supported.observed_loss_kg - supported.predicted_loss_kg
        result.update(district_mae_kg=float(np.abs(error).mean()) if len(error) else np.nan,
                      district_rmse_kg=float(np.sqrt((error ** 2).mean())) if len(error) else np.nan,
                      total_predicted_loss_kg=float(agg.predicted_loss_kg.sum()),
                      total_observed_loss_kg=float(agg.observed_loss_kg.sum()))
    return result


class HistoricalRate:
    """Exact prior-period means or empirical shrinkage toward broader groups."""
    LEVELS = [("crop_name",), ("crop_name", "season"), ("district", "crop_name"), ("district", "crop_name", "season")]

    def __init__(self, alpha=0):
        self.alpha = float(alpha)

    def fit(self, train):
        y = train[TARGET].to_numpy(float)
        self.mean_ = float(y.mean())
        self.prob_ = float((y > 0).mean())
        self.tables_ = {}
        z = train.assign(_positive=(y > 0).astype(int))
        for keys in self.LEVELS:
            self.tables_[keys] = z.groupby(list(keys), as_index=False).agg(_sum=(TARGET, "sum"), _n=(TARGET, "size"), _pos=("_positive", "sum"))
        return self

    def _lookup(self, frame, keys, parent_rate, parent_prob):
        joined = frame[list(keys)].merge(self.tables_[keys], on=list(keys), how="left", sort=False)
        n = joined._n.fillna(0).to_numpy()
        denominator = n + self.alpha
        rate = np.divide(joined._sum.fillna(0).to_numpy() + self.alpha * parent_rate, denominator,
                         out=np.asarray(parent_rate, float).copy(), where=denominator > 0)
        prob = np.divide(joined._pos.fillna(0).to_numpy() + self.alpha * parent_prob, denominator,
                         out=np.asarray(parent_prob, float).copy(), where=denominator > 0)
        return rate, prob, n > 0

    def predict_components(self, frame):
        count = len(frame)
        base, pbase = np.full(count, self.mean_), np.full(count, self.prob_)
        crop, pcrop, _ = self._lookup(frame, self.LEVELS[0], base, pbase)
        cs, pcs, cs_known = self._lookup(frame, self.LEVELS[1], crop, pcrop)
        dc, pdc, dc_known = self._lookup(frame, self.LEVELS[2], crop, pcrop)
        if self.alpha:
            parent, pparent = .5 * (cs + dc), .5 * (pcs + pdc)
        else:
            parent, pparent = np.where(dc_known, dc, cs), np.where(dc_known, pdc, pcs)
        rate, prob, _ = self._lookup(frame, self.LEVELS[3], parent, pparent)
        return np.clip(rate, 0, 100), np.clip(prob, 0, 1), np.divide(rate, prob, out=np.zeros(count), where=prob > 0)

    def predict(self, frame):
        return self.predict_components(frame)[0]


def preprocessor(features):
    nums = [c for c in features if c not in CATS]
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), CATS),
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)), nums),
    ])


def _holdout(frame, fraction, seed, entity_column=None):
    group = groups(frame, entity_column)
    if group.nunique() < 3:
        raise ValueError("Too few groups for independent early-stopping and calibration subsets.")
    for attempt in range(20):
        left, right = next(GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed + attempt).split(frame, groups=group))
        if frame.iloc[left][TARGET].gt(0).sum() >= 8 and frame.iloc[right][TARGET].gt(0).sum() >= 2:
            assert set(group.iloc[left]).isdisjoint(group.iloc[right])
            return left, right
    raise ValueError("Too few positive losses in independent training groups. More data or a simpler baseline is needed.")


def prepare_partition(train, valid, features, seed=SEED, entity_column=None):
    train = train.reset_index(drop=True)
    base_idx, cal_idx = _holdout(train, .15, seed, entity_column)
    base, cal = train.iloc[base_idx].reset_index(drop=True), train.iloc[cal_idx].reset_index(drop=True)
    core_idx, stop_idx = _holdout(base, .15, seed + 100, entity_column)
    core, stop = base.iloc[core_idx], base.iloc[stop_idx]
    core_pp, base_pp = preprocessor(features), preprocessor(features)
    core_x = core_pp.fit_transform(core[features])
    base_x = base_pp.fit_transform(base[features])
    return {"base": base, "cal": cal, "core": core, "stop": stop, "valid": valid,
            "pp": base_pp, "core_x": core_x, "stop_x": core_pp.transform(stop[features]),
            "base_x": base_x, "cal_x": base_pp.transform(cal[features])}


def group_weights(frame, mode):
    if mode == "profiles":
        return np.ones(len(frame))
    size = frame.groupby(KEYS)[TARGET].transform("size").to_numpy(float)
    w = 1 / size
    return np.clip(w / w.mean(), .25, 4.0)


class ConstantEstimator:
    def __init__(self, value):
        self.value = float(value)

    def predict(self, x):
        return np.full(x.shape[0], self.value)

    def predict_proba(self, x):
        p = self.predict(x)
        return np.column_stack([1 - p, p])


def _fit_booster(part, params, stage, max_trees=500, threads=4):
    classification = stage == "occurrence"
    positive_only = stage == "severity"
    core_y, stop_y, base_y = [part[k][TARGET].to_numpy(float) for k in ["core", "stop", "base"]]
    masks = [(y > 0) if positive_only else np.ones(len(y), dtype=bool) for y in [core_y, stop_y, base_y]]
    if classification:
        core_y, stop_y, base_y = [(y > 0).astype(int) for y in [core_y, stop_y, base_y]]
        if np.unique(core_y).size < 2 or np.unique(base_y).size < 2:
            return ConstantEstimator(base_y.mean()), 0
    if masks[0].sum() < 2 or masks[2].sum() < 2:
        return ConstantEstimator(base_y[masks[2]].mean() if masks[2].any() else 0), 0
    objective = "binary:logistic" if classification else (
        "reg:gamma" if positive_only and params["severity_objective"] == "gamma" else "reg:tweedie")
    power = params.get("severity_power", 1.5) if positive_only else params.get("tweedie_power", 1.5)
    common = dict(objective=objective, tree_method="hist", max_bin=128, n_jobs=threads, random_state=SEED,
                  learning_rate=params["learning_rate"], max_depth=params["clf_depth"] if classification else params["reg_depth"],
                  min_child_weight=params["clf_min_child"] if classification else params["reg_min_child"],
                  subsample=params["subsample"], colsample_bytree=params["colsample"],
                  reg_alpha=params["reg_alpha"], reg_lambda=params["reg_lambda"], verbosity=0)
    if objective == "reg:tweedie":
        common["tweedie_variance_power"] = power
    common["eval_metric"] = "logloss" if classification else ("gamma-nloglik" if objective == "reg:gamma" else f"tweedie-nloglik@{power}")
    estimator = XGBClassifier if classification else XGBRegressor
    core_w = group_weights(part["core"], params["weight_mode"])[masks[0]]
    base_w = group_weights(part["base"], params["weight_mode"])[masks[2]]
    if masks[1].any():
        early = estimator(n_estimators=max_trees, early_stopping_rounds=30, **common)
        early.fit(part["core_x"][masks[0]], core_y[masks[0]], sample_weight=core_w,
                  eval_set=[(part["stop_x"][masks[1]], stop_y[masks[1]])], verbose=False)
        trees = int(early.best_iteration) + 1
    else:
        trees = min(100, max_trees)
    final = estimator(n_estimators=max(1, trees), **common)
    final.fit(part["base_x"][masks[2]], base_y[masks[2]], sample_weight=base_w, verbose=False)
    return final, trees


class RateModel:
    def __init__(self, pp, features, family):
        self.pp, self.features, self.family = pp, features, family
        self.calibrator, self.severity_scale = None, 1.0

    def predict_components(self, frame):
        if not len(frame):
            return np.array([]), np.array([]), np.array([])
        x = self.pp.transform(frame[self.features])
        if self.family == "tweedie":
            rate = np.clip(self.regressor.predict(x) * self.severity_scale, 0, 100)
            return rate, np.full(len(frame), np.nan), np.full(len(frame), np.nan)
        prob = self.classifier.predict_proba(x)[:, 1]
        if self.calibrator is not None:
            prob = self.calibrator.predict_proba(logit(np.clip(prob, 1e-6, 1 - 1e-6)).reshape(-1, 1))[:, 1]
        severity = np.clip(self.regressor.predict(x) * self.severity_scale, 0, 100)
        return np.clip(prob * severity, 0, 100), prob, severity

    def predict(self, frame):
        return self.predict_components(frame)[0]


def fit_rate_model(part, features, params, max_trees=500, threads=4):
    model = RateModel(part["pp"], features, params["family"])
    model.diagnostics = {"base_rows": len(part["base"]), "calibration_rows": len(part["cal"])}
    if params["family"] == "hurdle":
        model.classifier, trees = _fit_booster(part, params, "occurrence", max_trees, threads)
        model.diagnostics["classifier_trees"] = trees
        model.regressor, trees = _fit_booster(part, params, "severity", max_trees, threads)
        model.diagnostics["severity_trees"] = trees
    else:
        model.regressor, trees = _fit_booster(part, params, "combined", max_trees, threads)
        model.diagnostics["regression_trees"] = trees
    if params["calibration"] == "calibrated":
        y = part["cal"][TARGET].to_numpy(float)
        x = part["cal_x"]
        if params["family"] == "hurdle":
            positive = y > 0
            if positive.sum() >= 20 and (~positive).sum() >= 20:
                probability = model.classifier.predict_proba(x)[:, 1]
                model.calibrator = LogisticRegression(C=1.0, max_iter=500).fit(logit(np.clip(probability, 1e-6, 1 - 1e-6)).reshape(-1, 1), positive.astype(int))
                denominator = np.maximum(model.regressor.predict(x[positive]), 1e-8).sum()
                model.severity_scale = float(np.clip(y[positive].sum() / denominator, .5, 2))
                model.diagnostics["calibration"] = "held-out sigmoid occurrence and positive-mean severity scaling"
            else:
                model.diagnostics["calibration"] = "skipped: insufficient positive/negative calibration cases"
        else:
            denominator = np.maximum(model.regressor.predict(x), 1e-8).sum()
            model.severity_scale = float(np.clip(y.sum() / denominator, .5, 2))
            model.diagnostics["calibration"] = "held-out mean scaling"
    return model


def baseline_oof(dev, splits, alpha=0):
    pred = np.zeros(len(dev))
    for train, valid in splits:
        pred[valid] = HistoricalRate(alpha).fit(dev.iloc[train]).predict(dev.iloc[valid])
    return pred


def default_params():
    return dict(family="hurdle", severity_objective="gamma", severity_power=1.5, tweedie_power=1.5,
                learning_rate=.05, clf_depth=3, reg_depth=3, clf_min_child=5., reg_min_child=1.,
                subsample=.9, colsample=.9, reg_alpha=.01, reg_lambda=5., weight_mode="profiles", calibration="calibrated")


def cv_predict(dev, partitions, splits, features, params, max_trees=500, threads=4):
    out, prob = np.zeros(len(dev)), np.full(len(dev), np.nan)
    for part, (_, valid) in zip(partitions, splits):
        fitted = fit_rate_model(part, features, params, max_trees, threads)
        out[valid], prob[valid], _ = fitted.predict_components(dev.iloc[valid])
    return out, prob


def fingerprint(dev, features, config):
    digest = hashlib.sha256(pd.util.hash_pandas_object(dev, index=True).to_numpy().tobytes())
    digest.update(json.dumps([CODE_VERSION, features, config, versions()], sort_keys=True).encode())
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def search_models(dev, features, output_dir, n_trials=12, max_trees=500, n_splits=3, min_profiles=10, threads=4, entity_column=None):
    import optuna
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    quantity = "production_kg" in dev
    error_metric = "district_mae_kg" if quantity else "district_mae_pp"
    config = dict(max_trees=max_trees, n_splits=n_splits, min_profiles=min_profiles, entity_column=entity_column,
                  ranking_basis="predicted_loss_kg" if quantity else "predicted_loss_pct",
                  loss_quantity_scope="survey_expanded" if "survey_weight" in dev else "sample_total" if quantity else "unavailable")
    stamp = fingerprint(dev, features, config)
    splits = cv_splits(dev, n_splits, entity_column)
    baseline_predictions, baseline_rows = {}, []
    for alpha in [0, 50, 200]:
        name = "Historical exact" if alpha == 0 else f"Historical shrinkage {alpha}"
        pred = baseline_oof(dev, splits, alpha)
        baseline_predictions[name] = pred
        baseline_rows.append({"model": name, "alpha": alpha, **metrics(dev, pred, min_profiles)})
    zero = metrics(dev, np.zeros(len(dev)), min_profiles)
    baseline_rows.append({"model": "Always zero (diagnostic only)", "alpha": np.nan, **zero})
    baselines = pd.DataFrame(baseline_rows)
    baselines.to_csv(output_dir / "baseline_cv_2024.csv", index=False)
    if not baselines.ranking_cohorts.max():
        raise ValueError("No crop-season cohort has enough supported districts and nonconstant outcomes. Ranking selection is not possible.")
    print("Preparing independent early-stopping/calibration subsets (2024 only)...", flush=True)
    partitions = [prepare_partition(dev.iloc[tr], dev.iloc[va], features, SEED + i, entity_column) for i, (tr, va) in enumerate(splits)]
    state_path = output_dir / "search_checkpoint.joblib"
    study = None
    if state_path.exists():
        saved = joblib.load(state_path)  # Load only your own trusted checkpoint.
        if saved.get("fingerprint") == stamp and saved.get("optuna_version") == optuna.__version__:
            study = saved["study"]
            print("Resuming compatible 2024-only search.", flush=True)
    if study is None:
        startup = min(4, max(1, n_trials // 3))
        study = optuna.create_study(directions=["maximize", "minimize"], sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=startup))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cache_dir = output_dir / ("oof_" + stamp[:12])
    cache_dir.mkdir(exist_ok=True)

    def objective(trial):
        started = time.monotonic()
        params = default_params()
        params.update(family=trial.suggest_categorical("family", ["hurdle", "tweedie"]),
                      calibration=trial.suggest_categorical("calibration", ["none", "calibrated"]),
                      learning_rate=trial.suggest_float("learning_rate", .025, .12, log=True),
                      clf_depth=trial.suggest_int("clf_depth", 2, 4), reg_depth=trial.suggest_int("reg_depth", 2, 4),
                      clf_min_child=trial.suggest_float("clf_min_child", 2, 30, log=True),
                      reg_min_child=trial.suggest_float("reg_min_child", .1, 8, log=True),
                      subsample=trial.suggest_float("subsample", .75, 1), colsample=trial.suggest_float("colsample", .75, 1),
                      reg_alpha=trial.suggest_float("reg_alpha", 1e-4, .5, log=True),
                      reg_lambda=trial.suggest_float("reg_lambda", 1, 30, log=True),
                      weight_mode=trial.suggest_categorical("weight_mode", ["profiles", "district_balanced"]))
        if params["family"] == "hurdle":
            params["severity_objective"] = trial.suggest_categorical("severity_objective", ["gamma", "tweedie"])
            params["severity_power"] = trial.suggest_float("severity_power", 1.1, 1.8)
        else:
            params["tweedie_power"] = trial.suggest_float("tweedie_power", 1.1, 1.8)
        prediction, probability = cv_predict(dev, partitions, splits, features, params, max_trees, threads)
        result = metrics(dev, prediction, min_profiles)
        if not np.isfinite(result["ndcg5_macro"]):
            raise ValueError("No ranking metric can be evaluated on these folds.")
        trial.set_user_attr("resolved_params", params)
        trial.set_user_attr("metrics", {key: value if np.isfinite(value) else None for key, value in result.items()})
        np.savez_compressed(cache_dir / f"trial_{trial.number}.npz", prediction=prediction, probability=probability)
        print(f"Trial {trial.number + 1}: NDCG@5={result['ndcg5_macro']:.3f}, {error_metric}={result[error_metric]:.3f}, {time.monotonic() - started:.1f}s", flush=True)
        return result["ndcg5_macro"], result[error_metric]

    def checkpoint(study, trial):
        temporary = state_path.with_suffix(".tmp")
        joblib.dump({"fingerprint": stamp, "study": study, "optuna_version": optuna.__version__}, temporary)
        os.replace(temporary, state_path)
        study.trials_dataframe().to_csv(output_dir / "tuning_trials_2024.csv", index=False)

    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    study.optimize(objective, n_trials=max(0, n_trials - len(complete)), callbacks=[checkpoint], n_jobs=1)
    # OOF scores here are selection scores, not an unbiased nested-CV estimate.
    rows = baseline_rows[:-1].copy()
    choices = {row["model"]: {"kind": "baseline", "alpha": row["alpha"]} for row in rows}
    predictions = dict(baseline_predictions)
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        path = cache_dir / f"trial_{trial.number}.npz"
        if path.exists():
            pred = np.load(path)["prediction"]
        else:
            pred, _ = cv_predict(dev, partitions, splits, features, trial.user_attrs["resolved_params"], max_trees, threads)
        for alpha in [0, 50, 200]:
            base_name = "Historical exact" if alpha == 0 else f"Historical shrinkage {alpha}"
            for weight in [.25, .5, .75, 1.]:
                if weight == 1 and alpha != 0:
                    continue
                name = f"Trial {trial.number}: ML={weight:g}, prior alpha={alpha}"
                blended = weight * pred + (1 - weight) * baseline_predictions[base_name]
                rows.append({"model": name, **metrics(dev, blended, min_profiles)})
                predictions[name] = blended
                choices[name] = {"kind": "blend", "ml_weight": weight, "alpha": alpha, "params": trial.user_attrs["resolved_params"]}
    comparison = pd.DataFrame(rows)
    reference = baselines.loc[baselines.model.ne("Always zero (diagnostic only)"), error_metric].min()
    admissible = comparison.loc[comparison[error_metric] <= reference * 1.10 + 1e-12]
    winner = admissible.sort_values(["ndcg5_macro", error_metric, "model"], ascending=[False, True, True]).iloc[0]
    choice = choices[winner.model]
    lock = {"choice": choice, "features": features, "fingerprint": stamp, "config": config,
            "selection": winner.to_dict(), "selection_rule": f"2024 OOF: best macro NDCG@5 ranked by {config['ranking_basis']}, with {error_metric} <= 1.10 times the best historical baseline; zero predictor diagnostic only",
            "versions": versions(), "test_year": 2025, "new_unseen_year_still_required": True}
    (output_dir / "locked_choice_before_2025.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    comparison.to_csv(output_dir / "candidate_comparison_2024.csv", index=False)
    return lock, predictions[winner.model], comparison


class BlendedRate:
    def __init__(self, prior, ml=None, ml_weight=0):
        self.prior, self.ml, self.ml_weight = prior, ml, ml_weight

    def predict_components(self, frame):
        rate, prob, severity = self.prior.predict_components(frame)
        if self.ml is None or self.ml_weight == 0:
            return rate, prob, severity
        other, other_prob, _ = self.ml.predict_components(frame)
        rate = self.ml_weight * other + (1 - self.ml_weight) * rate
        if np.isfinite(other_prob).all():
            prob = self.ml_weight * other_prob + (1 - self.ml_weight) * prob
            severity = np.divide(rate, prob, out=np.zeros(len(frame)), where=prob > 0)
        else:
            prob, severity = np.full(len(frame), np.nan), np.full(len(frame), np.nan)
        return np.clip(rate, 0, 100), prob, severity

    def predict(self, frame):
        return self.predict_components(frame)[0]


def fit_locked(dev, lock, threads=4):
    if not dev.year.eq(2024).all():
        raise ValueError("The evaluation model must fit 2024 only.")
    choice, config, features = lock["choice"], lock["config"], lock["features"]
    prior = HistoricalRate(choice["alpha"]).fit(dev)
    if choice["kind"] == "baseline":
        return BlendedRate(prior)
    part = prepare_partition(dev, None, features, SEED + 900, config["entity_column"])
    ml = fit_rate_model(part, features, choice["params"], config["max_trees"], threads)
    return BlendedRate(prior, ml, choice["ml_weight"])


def component_metrics(frame, rate, probability, severity):
    y = frame[TARGET].to_numpy(float)
    result = {"rows": len(y), "positive_cases": int((y > 0).sum()), "positive_prevalence": float((y > 0).mean())}
    if np.isfinite(probability).all():
        result["occurrence_brier"] = float(brier_score_loss(y > 0, probability))
        result["occurrence_average_precision"] = float(average_precision_score(y > 0, probability)) if (y > 0).any() else np.nan
    if (y > 0).any() and np.isfinite(severity).all():
        result["conditional_positive_rmse_pp"] = float(np.sqrt(np.mean((severity[y > 0] - y[y > 0]) ** 2)))
        result["conditional_mean_predicted_pct"] = float(severity[y > 0].mean())
        result["conditional_mean_observed_pct"] = float(y[y > 0].mean())
    return result


def evaluate_locked(dev, test, bundle, lock, output_dir):
    if not test.year.eq(2025).all():
        raise ValueError("Temporal evaluation expects 2025 rows only.")
    output_dir = Path(output_dir)
    min_profiles = lock["config"]["min_profiles"]
    ranking_basis = "predicted_loss_kg" if "production_kg" in test else "predicted_loss_pct"
    scope = "survey_expanded" if "survey_weight" in test else "sample_total" if "production_kg" in test else "unavailable"
    if ranking_basis != lock["config"].get("ranking_basis", "predicted_loss_pct") or scope != lock["config"].get("loss_quantity_scope", "unavailable"):
        raise ValueError("Temporal evaluation must use the locked ranking basis and quantity scope; do not mix rates, sample totals and survey-expanded totals.")
    rate, probability, severity = bundle.predict_components(test)
    predictors = {"V3 locked candidate": rate,
                  "Historical exact": HistoricalRate(0).fit(dev).predict(test),
                  "Historical shrinkage 50": HistoricalRate(50).fit(dev).predict(test),
                  "Always zero (diagnostic only)": np.zeros(len(test))}
    rows = []
    for scope, mask in [("2025 all", np.ones(len(test), dtype=bool)), ("2025 A+B", test.season.isin(["A", "B"]).to_numpy())]:
        if mask.any():
            for name, values in predictors.items():
                rows.append({"scope": scope, "model": name, **metrics(test.loc[mask], values[mask], min_profiles)})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "external_validation_2025.csv", index=False)
    seasons, components = [], []
    for season in sorted(test.season.unique()):
        mask = test.season.eq(season).to_numpy()
        seasons.append({"season": season, "positive_cases": int(test.loc[mask, TARGET].gt(0).sum()), **metrics(test.loc[mask], rate[mask], min_profiles)})
        components.append({"season": season, **component_metrics(test.loc[mask], rate[mask], probability[mask], severity[mask])})
    pd.DataFrame(seasons).to_csv(output_dir / "external_validation_by_season_2025.csv", index=False)
    pd.DataFrame(components).to_csv(output_dir / "component_diagnostics_2025.csv", index=False)
    by_crop = [{"crop_name": crop, **metrics(part, rate[test.crop_name.eq(crop).to_numpy()], min_profiles)} for crop, part in test.groupby("crop_name")]
    pd.DataFrame(by_crop).to_csv(output_dir / "external_validation_by_crop_2025.csv", index=False)
    agg = aggregate_rates(test, rate)
    risk = agg.rename(columns={"predicted": "predicted_loss_pct", "observed": "observed_loss_pct", "profiles": "n_profiles"})
    risk["vulnerability_rank"] = risk.groupby(["season", "crop_name"])[ranking_basis].rank(method="min", ascending=False).astype(int)
    risk["support_level"] = pd.cut(risk.n_profiles, [-1, 9, 29, 99, np.inf], labels=["Limited", "Low", "Moderate", "Strong"]).astype(str)
    risk.to_csv(output_dir / "harvestguard_2025_v3_vulnerability.csv", index=False)
    ranking_table(agg, min_profiles).to_csv(output_dir / "ranking_cohorts_2025.csv", index=False)
    for name, prediction in predictors.items():
        if name in {"V3 locked candidate", "Historical exact"}:
            detail = ranking_table(aggregate_rates(test, prediction), min_profiles)
            detail.to_csv(output_dir / ("ranking_detail_candidate.csv" if name == "V3 locked candidate" else "ranking_detail_baseline.csv"), index=False)
    profile = test[KEYS + [TARGET]].copy()
    profile["expected_loss_pct"], profile["occurrence_probability"], profile["conditional_severity_pct"] = rate, probability, severity
    if "production_kg" in test:
        quantities = profile_loss_quantities(test, rate)
        for name in ["production_kg", "survey_weight", "predicted_loss_kg", "observed_loss_kg", "weighted_production_kg", "weighted_predicted_loss_kg", "weighted_observed_loss_kg"]:
            profile[name] = quantities[name].to_numpy()
    profile.to_csv(output_dir / "profile_predictions_2025.csv", index=False)
    calibration = pd.DataFrame({"observed": test[TARGET].to_numpy(), "predicted": rate})
    calibration["bin"] = pd.qcut(calibration.predicted, 10, duplicates="drop") if np.unique(rate).size > 1 else "constant prediction"
    cal = calibration.groupby("bin", observed=True).agg(n=("observed", "size"), mean_predicted_pct=("predicted", "mean"), mean_observed_pct=("observed", "mean")).reset_index()
    cal.to_csv(output_dir / "calibration_2025.csv", index=False)
    ab = comparison.loc[comparison.scope.eq("2025 A+B")].set_index("model")
    error_metric = "district_mae_kg" if "production_kg" in test else "district_mae_pp"
    checks = {"A+B district MAE beats exact history": False, "A+B NDCG beats exact history": False,
              "A+B top5 overlap beats exact history": False}
    if len(ab):
        candidate, baseline = ab.loc["V3 locked candidate"], ab.loc["Historical exact"]
        checks = {"A+B district MAE beats exact history": bool(candidate[error_metric] < baseline[error_metric]),
                  "A+B NDCG beats exact history": bool(candidate.ndcg5_macro > baseline.ndcg5_macro),
                  "A+B top5 overlap beats exact history": bool(candidate.top5_overlap_macro > baseline.top5_overlap_macro)}
    pd.DataFrame([{"check": k, "passed": v} for k, v in checks.items()]).to_csv(output_dir / "readiness_checks.csv", index=False)
    return comparison, risk, cal, checks


def drift_report(dev, test, features):
    rows = []
    for name in features:
        if name in CATS:
            rows.append({"feature": name, "kind": "category", "unseen_2025_fraction": float((~test[name].isin(dev[name].unique())).mean())})
        else:
            std = dev[name].std()
            rows.append({"feature": name, "kind": "numeric", "mean_2024": dev[name].mean(), "mean_2025": test[name].mean(),
                         "standardized_mean_shift": (test[name].mean() - dev[name].mean()) / std if pd.notna(std) and std > 0 else np.nan,
                         "missing_2024_fraction": dev[name].isna().mean(), "missing_2025_fraction": test[name].isna().mean()})
    return pd.DataFrame(rows)
