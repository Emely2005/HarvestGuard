"""Export district evidence from the existing model; never train or tune it.

Run in the saved model environment, then serve the small aggregate export in the
dashboard. Farmer records stay local. Numeric sensitivities use a documented
2024 crop/season reference; they are not SHAP values or intervention effects.
"""

import argparse
import calendar
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import harvestguard_model_v3 as hg


LABELS = {
    "season": "Season", "district": "District", "crop_name": "Crop",
    "farmer_type": "Farmer type", "harvest_month": "Harvest month",
    "n_plots_crop": "Number of crop plots", "crop_plot_area_ha_total": "Crop plot area",
    "num_crops_in_plot": "Crops per plot", "rain_30d_before_mm": "Rainfall before harvest",
    "humidity_mean_7d_before_pct": "Humidity before harvest",
}
KEYS = ["season", "crop_name", "district"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_evidence(input_path, model_dir, quantity_dir, output_dir):
    model_path = model_dir / "model_v3_candidate.joblib"
    model_hash = digest(model_path)
    manifest = json.loads((quantity_dir / "quantity_aggregation_metadata.json").read_text(encoding="utf-8"))
    if manifest["source_model_sha256"] != model_hash or manifest["input_sha256"] != digest(input_path):
        raise ValueError("Insight evidence must use the model and input that produced the quantity results.")
    if manifest.get("survey_weight_source_column"):
        raise ValueError("This exporter currently supports the verified sample_total results only.")
    metadata = json.loads((model_dir / "model_v3_metadata.json").read_text(encoding="utf-8"))
    raw = pd.read_csv(input_path)
    frame = hg.prepare_features(raw, metadata["features"])
    selected = frame.loc[raw.year.eq(2025) & frame.crop_name.isin(metadata["eligible_crops"])].copy()
    selected["production_kg"] = pd.to_numeric(selected[manifest["production_source_column"]], errors="raise")
    if not np.isfinite(selected.production_kg).all() or not selected.production_kg.gt(0).all():
        raise ValueError("Production must be finite positive kg.")
    model = hg.load_saved_rate_model(model_path)
    if not np.isclose(model.ml_weight, 1):
        raise ValueError("Component importance export requires this saved pure-ML candidate.")
    features = metadata["features"]
    rates = np.asarray(model.predict(selected[features]), float)
    selected["loss_kg"] = rates / 100 * selected.production_kg
    totals = selected.groupby(KEYS).agg(loss_kg=("loss_kg", "sum"), production_kg=("production_kg", "sum"), n_profiles=("loss_kg", "size"))
    saved = pd.read_csv(quantity_dir / "harvestguard_2025_v3_quantity_priorities.csv").set_index(KEYS)
    aligned = saved.reindex(totals.index)
    if len(totals) != len(saved) or not np.allclose(totals.loss_kg, aligned.predicted_loss_kg, atol=1e-3, rtol=1e-7):
        raise ValueError("Recomputed quantities do not match the dashboard results.")
    print(f"Verified {len(selected):,} profiles and {len(totals):,} saved quantities.", flush=True)

    # Sum gain over encoded categories to report original model features.
    encoded = list(model.ml.pp.get_feature_names_out())
    owners = []
    for name in encoded:
        owner = next((feature for feature in sorted(features, key=len, reverse=True)
                      if name.split("__", 1)[-1] == feature or name.split("__", 1)[-1].startswith(feature + "_")), None)
        if owner is None:
            raise ValueError(f"Cannot map encoded feature {name} to an original input.")
        owners.append(owner)
    importance = pd.DataFrame({"feature": features})
    for component, estimator in (("occurrence", model.ml.classifier), ("severity", model.ml.regressor)):
        scores = estimator.get_booster().get_score(importance_type="total_gain")
        grouped = {feature: 0.0 for feature in features}
        for index, feature in enumerate(owners):
            grouped[feature] += scores.get(f"f{index}", scores.get(encoded[index], 0.0))
        total = sum(grouped.values())
        importance[component + "_importance"] = importance.feature.map(grouped) / total
    importance["combined_importance"] = importance[["occurrence_importance", "severity_importance"]].mean(axis=1)
    importance["label"] = importance.feature.map(LABELS)
    importance = importance.sort_values("combined_importance", ascending=False).reset_index(drop=True)

    # Re-score unchanged profiles with one numeric input replaced at a time.
    # The reference is derived solely from development covariates, never labels.
    development = frame.loc[raw.year.eq(2024)]
    references = development.groupby(["crop_name", "season"])[hg.BEFORE].median()
    ref_index = pd.MultiIndex.from_frame(selected[["crop_name", "season"]])
    for feature in hg.BEFORE:
        reference = references[feature].reindex(ref_index).to_numpy(float)
        reference = np.where(np.isfinite(reference), reference, development[feature].median())
        counter = selected[features].copy()
        counter[feature] = reference
        difference = rates - np.asarray(model.predict(counter), float)
        selected[feature + "_delta_kg"] = difference / 100 * selected.production_kg
        selected[feature + "_reference"] = reference
        print(f"Exported local sensitivity: {feature}.", flush=True)

    records = []
    for key, group in selected.groupby(KEYS, sort=True):
        production = float(group.production_kg.sum())
        loss = float(group.loss_kg.sum())
        record = {"district": key[2], "crop": key[1], "season": key[0], "year": 2025,
                  "n_profiles": len(group), "production_kg": production, "expected_loss_kg": loss,
                  "expected_loss_pct": 100 * loss / production, "drivers": [], "weather": {}}
        # Suppress detailed cohort summaries for small groups.
        if len(group) >= 10:
            for feature in hg.BEFORE:
                valid = group[feature].notna()
                actual = float(np.average(group.loc[valid, feature], weights=group.loc[valid, "production_kg"])) if valid.any() else None
                reference = float(np.average(group[feature + "_reference"], weights=group.production_kg))
                delta = float(group[feature + "_delta_kg"].sum())
                record["drivers"].append({"feature": feature, "label": LABELS[feature],
                    "component": "expected_loss", "contribution": 100 * delta / production,
                    "unit": "percentage-point sensitivity to the development reference",
                    "delta_kg": delta, "actual_mean": actual, "reference_value": reference,
                    "coverage_pct": float(100 * group.loc[valid, "production_kg"].sum() / production)})
                if feature in {"humidity_mean_7d_before_pct", "rain_30d_before_mm"}:
                    record["weather"][feature] = {"value": actual, "coverage_pct": record["drivers"][-1]["coverage_pct"]}
            month_summary = group.groupby("harvest_month").agg(loss_kg=("loss_kg", "sum"), count=("loss_kg", "size"))
            month_summary = month_summary.loc[month_summary["count"].ge(5)]
            if loss > 0 and not month_summary.empty:
                month = month_summary.sort_values("loss_kg", ascending=False).index[0]
                try:
                    month_name = calendar.month_name[int(float(month))]
                except (ValueError, IndexError):
                    month_name = ""
                if month_name:
                    record["timing"] = {"month": month_name, "loss_kg": float(month_summary.loc[month, "loss_kg"]),
                                        "loss_share_pct": float(100 * month_summary.loc[month, "loss_kg"] / loss)}
            if len(group) >= 25 and loss > 0:
                count = int(np.ceil(.2 * len(group)))
                largest = group.sort_values("production_kg", ascending=False, kind="stable").head(count)
                record["concentration"] = {"profile_share_pct": 100 * count / len(group),
                    "loss_share_pct": float(100 * largest.loss_kg.sum() / loss),
                    "loss_kg": float(largest.loss_kg.sum()), "count": count}
        record["drivers"].sort(key=lambda item: abs(item["contribution"]), reverse=True)
        records.append(record)
    # Median across district means, within the same crop and season.
    for record in records:
        for feature, weather in record["weather"].items():
            peers = [r["weather"][feature]["value"] for r in records if r["crop"] == record["crop"] and r["season"] == record["season"]
                     and feature in r["weather"] and r["weather"][feature]["value"] is not None]
            weather["peer_median"] = float(np.median(peers))

    output_dir.mkdir(parents=True, exist_ok=True)
    importance.to_csv(output_dir / "feature_importance_hurdle.csv", index=False)
    export = {"schema_version": 1, "evidence_version": "insights-1", "source_model_sha256": model_hash,
              "input_sha256": manifest["input_sha256"], "quantity_csv_sha256": digest(quantity_dir / "harvestguard_2025_v3_quantity_priorities.csv"),
              "model_retrained": False, "features": features, "profiles": len(selected),
              "importance_csv_sha256": digest(output_dir / "feature_importance_hurdle.csv"),
              "method": "One-variable development-reference sensitivity of the complete calibrated hurdle model",
              "aggregation": "Sum profile-level sensitivity kg, then divide by the same group's production kg for percentage points",
              "reference": "2024 crop-season medians; 2024 overall feature median only for missing crop-season references",
              "importance_method": "Original-feature summed total_gain, normalised separately per hurdle component; combined score is the equal-weight mean of the two normalised scores",
              "limitations": ["Sensitivity is a model comparison, not a causal effect, SHAP decomposition or estimated intervention benefit.",
                  "Changing one input can break correlations; sensitivities must not be summed or interpreted as kg that an action will save.",
                  "Weather is measured before harvest; it does not measure grain moisture or storage conditions.",
                  "Detailed group summaries require at least ten profiles; timing cells require at least five; concentration requires at least twenty-five.",
                  "Largest-producer cohort membership is determined by production alone, with original row order resolving ties."],
              "records": records}
    if digest(model_path) != model_hash:
        raise ValueError("The source model changed during export.")
    (output_dir / "district_evidence.json").write_text(json.dumps(export, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved aggregate evidence to {output_dir}; model unchanged.", flush=True)
    print(importance[["label", "combined_importance"]].head(5).to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("postharvest_regression_model_ready_2024_2025_with_production.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("model_v3_candidate"))
    parser.add_argument("--quantity-dir", type=Path, default=Path("model_v3_quantity_results"))
    parser.add_argument("--output-dir", type=Path, default=Path("model_v3_insights"))
    args = parser.parse_args()
    export_evidence(args.input, args.model_dir, args.quantity_dir, args.output_dir)
