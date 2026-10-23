"""Load the V3 candidate's saved dashboard exports without unpickling the model."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_FOLDER = "model_v3_candidate"
PREDICTIONS_FILE = "harvestguard_2025_v3_vulnerability.csv"
METADATA_FILE = "model_v3_metadata.json"
QUANTITY_FOLDER = "model_v3_quantity_results"
QUANTITY_PREDICTIONS_FILE = "harvestguard_2025_v3_quantity_priorities.csv"
QUANTITY_METADATA_FILE = "quantity_aggregation_metadata.json"
REQUIRED_FILES = (
    PREDICTIONS_FILE, METADATA_FILE, "readiness_checks.csv",
    "external_validation_2025.csv", "external_validation_by_season_2025.csv",
    "model_v3_candidate.joblib", "harvestguard_model_v3.py",
)

QUANTITY_COLUMNS = {"production_kg", "predicted_loss_kg", "observed_loss_kg", "predicted_loss_tonnes", "observed_loss_tonnes", "loss_quantity_scope", "priority_rank"}


def rank_districts(view: pd.DataFrame) -> pd.DataFrame:
    """Rank kg totals when present; retain explicit rate-only legacy viewing."""
    column = "predicted_loss_kg" if "predicted_loss_kg" in view else "predicted_loss_pct"
    result = view.sort_values([column, "district"], ascending=[False, True]).reset_index(drop=True).copy()
    result["display_rank"] = result[column].rank(method="min", ascending=False).astype(int)
    return result


def validate_quantities(risk: pd.DataFrame, metadata: dict) -> None:
    """Reject partial/mixed quantity exports instead of silently ranking rates."""
    present = QUANTITY_COLUMNS.intersection(risk.columns)
    if not present:
        if metadata.get("production_burden_available") or metadata.get("ranking_basis") == "predicted_loss_kg":
            raise ValueError("Metadata promises kg priorities but the prediction file has no quantities.")
        return
    missing = QUANTITY_COLUMNS.difference(risk.columns)
    if missing:
        raise ValueError("Incomplete quantity export: " + ", ".join(sorted(missing)))
    for name in ("production_kg", "predicted_loss_kg", "observed_loss_kg", "priority_rank"):
        risk[name] = pd.to_numeric(risk[name], errors="raise")
        if not np.isfinite(risk[name]).all() or (risk[name] < 0).any():
            raise ValueError(f"{name} must contain finite nonnegative quantities.")
    if not risk.production_kg.gt(0).all():
        raise ValueError("Aggregated production_kg must be positive.")
    scopes = set(risk.loss_quantity_scope)
    if len(scopes) != 1 or not scopes.issubset({"sample_total", "survey_expanded"}):
        raise ValueError("Quantity scope must be uniformly sample_total or survey_expanded.")
    scope = next(iter(scopes))
    if metadata.get("loss_quantity_scope") != scope or metadata.get("production_burden_available") is not True or metadata.get("ranking_basis") != "predicted_loss_kg":
        raise ValueError("Quantity columns and metadata must agree on the ranking basis and scope.")
    if (scope == "survey_expanded") != (metadata.get("survey_weights_available") is True):
        raise ValueError("Survey-expanded quantities require matching survey-weight metadata.")
    for kind in ("predicted", "observed"):
        kg = risk[f"{kind}_loss_kg"]
        if (kg > risk.production_kg + 1e-8).any() or not np.allclose(100 * kg / risk.production_kg, risk[f"{kind}_loss_pct"], atol=1e-6, rtol=1e-6):
            raise ValueError(f"{kind} kg quantities disagree with the production-weighted percentages.")
        tonnes = f"{kind}_loss_tonnes"
        if tonnes in risk:
            risk[tonnes] = pd.to_numeric(risk[tonnes], errors="raise")
            if not np.allclose(risk[tonnes], kg / 1000, atol=1e-8, rtol=1e-6):
                raise ValueError(f"{tonnes} disagrees with kg quantities.")
    expected_rank = risk.groupby(["season", "crop_name"]).predicted_loss_kg.rank(method="min", ascending=False)
    if not risk.priority_rank.eq(expected_rank).all() or not risk.vulnerability_rank.eq(expected_rank).all():
        raise ValueError("Saved district ranks must be based on predicted_loss_kg, with equal ranks for ties.")


def bundle_signature(model_dir: Path) -> tuple:
    """Refresh Streamlit's data cache when the downloaded bundle changes."""
    signature = []
    for name in REQUIRED_FILES:
        path = model_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing V3 candidate file: {path.name}")
        stat = path.stat()
        signature.append((name, stat.st_mtime_ns, stat.st_size))
    return tuple(signature)


def _validate_prediction_table(risk: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    required = {
        "season", "district", "crop_name", "predicted_loss_pct", "observed_loss_pct",
        "n_profiles", "vulnerability_rank", "support_level",
    }
    missing = required.difference(risk.columns)
    if missing:
        raise ValueError("The V3 vulnerability file is missing columns: " + ", ".join(sorted(missing)))
    if risk.empty:
        raise ValueError("The V3 vulnerability file contains no predictions.")
    for name in ("season", "district", "crop_name", "support_level"):
        if risk[name].isna().any() or risk[name].astype(str).str.strip().eq("").any():
            raise ValueError(f"The V3 vulnerability file has missing {name} values.")
        risk[name] = risk[name].astype(str)
    if risk.duplicated(["season", "crop_name", "district"]).any():
        raise ValueError("The V3 vulnerability file contains duplicate district/crop/season predictions.")
    for name in ("predicted_loss_pct", "observed_loss_pct", "n_profiles", "vulnerability_rank"):
        risk[name] = pd.to_numeric(risk[name], errors="raise")
        if not np.isfinite(risk[name]).all():
            raise ValueError(f"The V3 vulnerability file contains non-finite {name} values.")
    for name in ("predicted_loss_pct", "observed_loss_pct"):
        if not risk[name].between(0, 100).all():
            raise ValueError(f"{name} must be between 0 and 100 percent.")
    for name in ("n_profiles", "vulnerability_rank"):
        if not ((risk[name] >= 1) & (risk[name] % 1 == 0)).all():
            raise ValueError(f"{name} must contain positive whole numbers.")
        risk[name] = risk[name].astype(int)
    validate_quantities(risk, metadata)
    return risk


def load_model_data(model_dir: Path) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    bundle_signature(model_dir)
    metadata = json.loads((model_dir / METADATA_FILE).read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or not metadata.get("candidate_version"):
        raise ValueError("The V3 metadata must identify its candidate_version.")
    if metadata.get("target") != "actionable_loss_rate_pct":
        raise ValueError("The V3 target must be actionable_loss_rate_pct (percent of production).")
    risk = _validate_prediction_table(pd.read_csv(model_dir / PREDICTIONS_FILE), metadata)
    metadata = {
        **metadata,
        "model_label": "Model V3 candidate",
        "model_version": metadata["candidate_version"],
        "prediction_file": PREDICTIONS_FILE,
        "validation_model": "V3 locked candidate",
        "baseline_model": "Historical exact",
        "cv_group": metadata.get("cv_grouping", "unspecified grouping"),
        "test_year": metadata.get("temporal_validation_year", 2025),
        "ranking_basis": "predicted_loss_kg" if "predicted_loss_kg" in risk else "predicted_loss_pct",
        "loss_quantity_scope": metadata.get("loss_quantity_scope", "unavailable"),
        "weather_note": (
            "The at-harvest feature policy uses rain in the preceding 30 days and humidity "
            "in the preceding 7 days. These measurements are not exported for each district."
            if metadata.get("feature_policy") == "at_harvest" else
            "Weather availability at prediction time needs review."
        ),
        "storage_method_note": "Storage method is not among the exported V3 features; do not claim it as a model driver.",
    }
    readiness = pd.read_csv(model_dir / "readiness_checks.csv")
    return risk, metadata, readiness


def _quantity_directory(model_dir: Path) -> Path | None:
    """Prefer the results folder, with support for a flat Colab download."""
    for directory in (model_dir.parent / QUANTITY_FOLDER, model_dir.parent):
        files = [directory / name for name in (QUANTITY_PREDICTIONS_FILE, QUANTITY_METADATA_FILE)]
        if any(path.exists() for path in files):
            missing = [path.name for path in files if not path.is_file()]
            if missing:
                raise FileNotFoundError("Incomplete quantity results: " + ", ".join(missing))
            return directory
    return None


def dashboard_signature(model_dir: Path) -> tuple:
    signature = list(bundle_signature(model_dir))
    directory = _quantity_directory(model_dir)
    if directory is not None:
        for name in (QUANTITY_PREDICTIONS_FILE, QUANTITY_METADATA_FILE):
            path = directory / name
            stat = path.stat()
            signature.append((str(path), stat.st_mtime_ns, stat.st_size))
    return tuple(signature)


def load_dashboard_data(model_dir: Path) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Use verified kg results while preserving the original model and reports."""
    risk, metadata, readiness = load_model_data(model_dir)
    if "predicted_loss_kg" in risk:
        return risk, metadata, readiness
    directory = _quantity_directory(model_dir)
    if directory is None:
        return risk, metadata, readiness
    manifest = json.loads((directory / QUANTITY_METADATA_FILE).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Quantity metadata must be an object.")
    if manifest.get("source_candidate_version") != metadata["candidate_version"] or manifest.get("target") != metadata["target"]:
        raise ValueError("Quantity results do not match the saved V3 candidate and target.")
    if manifest.get("year") != metadata["test_year"]:
        raise ValueError("Quantity results do not match the dashboard's historical year.")
    source_hash = hashlib.sha256((model_dir / "model_v3_candidate.joblib").read_bytes()).hexdigest()
    if manifest.get("source_model_sha256") != source_hash or manifest.get("source_model_unchanged") is not True:
        raise ValueError("Quantity results must identify the unchanged saved model by its SHA-256 hash.")
    if manifest.get("ranking_basis") != "predicted_loss_kg":
        raise ValueError("Quantity results must rank estimated kg lost.")
    if manifest.get("features") != metadata.get("features"):
        raise ValueError("Quantity results must use the saved model's feature definitions.")
    scope = manifest.get("loss_quantity_scope")
    combined = {
        **metadata,
        "production_burden_available": True,
        "ranking_basis": "predicted_loss_kg",
        "loss_quantity_scope": scope,
        "survey_weights_available": manifest.get("survey_weight_source_column") is not None,
        "prediction_file": (directory / QUANTITY_PREDICTIONS_FILE).relative_to(model_dir.parent).as_posix(),
        "quantity_source_metadata": (directory / QUANTITY_METADATA_FILE).relative_to(model_dir.parent).as_posix(),
        "production_source_column": manifest.get("production_source_column"),
        "production_source_unit": manifest.get("production_source_unit"),
        "validation_ranking_basis": metadata["ranking_basis"],
        "quantity_ranking_validation_recomputed": False,
        "model_ready_for_national_allocation": False,
    }
    quantities = _validate_prediction_table(pd.read_csv(directory / QUANTITY_PREDICTIONS_FILE), combined)
    if manifest.get("profiles_scored") != int(quantities.n_profiles.sum()):
        raise ValueError("Quantity results and metadata disagree on the number of scored profiles.")
    return quantities, combined, readiness
