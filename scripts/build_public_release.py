"""Package a public release from an explicit allow-list; never include raw data."""
import hashlib
import json
from pathlib import Path
import re
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "harvestguard_app.py", "dashboard.py", "harvestguard_ai.py", "harvestguard_ai_cache.py", "harvestguard_ai_view.py",
    "harvestguard_data.py", "harvestguard_decisions.py", "harvestguard_insights.py", "harvestguard_insights_export.py",
    "harvestguard_map.py", "harvestguard_ui.py", "harvestguard_views.py", "harvestguard_methodology.py", "harvestguard_model_v3.py",
    "harvestguard_groq.py", "harvestguard_chat.py", "harvestguard_chat_view.py",
    "README.md", "METHODOLOGY.md", "DEPLOYMENT.md", "SUBMISSION.md", "AI_SETUP.md", "MODEL_REVIEW.md", "PROJECT_AUDIT.md",
    "requirements.txt", "requirements-web.txt", "requirements-dev.txt", "Dockerfile", "render.yaml", ".dockerignore", ".gitignore",
    ".streamlit/config.toml", ".streamlit/secrets.toml.example",
    "ddi-documentation-english-113.pdf", "ddi-documentation-english-124.pdf",
    "HarvestGuard_Model_Development_V3_Colab.ipynb", "HarvestGuard_Quantity_Aggregation_Colab.ipynb",
    "model_v3_quantity_results/harvestguard_2025_v3_quantity_priorities.csv",
    "model_v3_quantity_results/quantity_aggregation_metadata.json", "model_v3_quantity_results/RUN_RESULTS.md",
    "model_v3_quantity_results/real_data_aggregation_checks.json",
    "model_v3_insights/district_evidence.json", "model_v3_insights/feature_importance_hurdle.csv", "model_v3_insights/RUN_RESULTS.md",
    "scripts/build_public_release.py", "scripts/audit_project.py",
]
# Every public artifact is named explicitly. Stray files in a retained directory
# must never be included automatically in a competition submission.
FILES += [
    'knowledge_base/district_context.json',
    'knowledge_base/interventions.json',
    'model_v3_candidate/baseline_cv_2024.csv',
    'model_v3_candidate/calibration_2025.csv',
    'model_v3_candidate/candidate_comparison_2024.csv',
    'model_v3_candidate/component_diagnostics_2025.csv',
    'model_v3_candidate/crop_eligibility_2024.csv',
    'model_v3_candidate/external_validation_2025.csv',
    'model_v3_candidate/external_validation_by_crop_2025.csv',
    'model_v3_candidate/external_validation_by_season_2025.csv',
    'model_v3_candidate/feature_drift_2024_to_2025.csv',
    'model_v3_candidate/harvestguard_2025_v3_vulnerability.csv',
    'model_v3_candidate/harvestguard_model_v3.py',
    'model_v3_candidate/locked_choice_before_2025.json',
    'model_v3_candidate/model_v3_candidate.joblib',
    'model_v3_candidate/model_v3_metadata.json',
    'model_v3_candidate/ranking_cohorts_2025.csv',
    'model_v3_candidate/ranking_detail_baseline.csv',
    'model_v3_candidate/readiness_checks.csv',
    'model_v3_candidate/README.md',
    'model_v3_candidate/requirements_model_v3.txt',
    'model_v3_candidate/target_by_year_season.csv',
    'model_v3_candidate/temporal_validation.png',
    'model_v3_candidate/tuning_trials_2024.csv',
    'source_preprocessing/build_postharvest_model_ready_corrected.py',
    'source_preprocessing/POSTHARVEST_CORRECTED_PREPROCESSING_README.txt',
    'source_preprocessing/postharvest_regression_model_dictionary.csv',
    'source_preprocessing/postharvest_regression_model_manifest.json',
    'source_preprocessing/postharvest_requirements.txt',
    'source_preprocessing/quality_metrics.json',
    'source_preprocessing/unified_postharvest_quality_report.txt',
    'tests/conftest.py',
    'tests/fixtures/legacy_v2/harvestguard_2025_v2_vulnerability.csv',
    'tests/fixtures/legacy_v2/model_v2_metadata.json',
    'tests/fixtures/legacy_v2/README.md',
    'tests/test_ai.py',
    'tests/test_app.py',
    'tests/test_dashboard_quantities.py',
    'tests/test_decisions.py',
    'tests/test_insight_planning.py',
    'tests/test_methodology.py',
    'tests/test_quantity_priorities.py',
    'tests/test_release.py',
    'tests/test_project_chat.py',
    'tests/test_v3_integration.py',
    'model_review_tests/requirements.txt',
    'model_review_tests/test_loss_quantities.py',
    'model_review_tests/test_model_v3.py',
]
FORBIDDEN_PARTS = {".git", ".runtime", ".aws", ".codex", ".agents", "__pycache__", ".pytest_cache", ".venv"}
SECRET = re.compile(r"AIza[0-9A-Za-z_-]{35}|gsk_[0-9A-Za-z]{20,}|sk-[0-9A-Za-z_-]{24,}|ghp_[0-9A-Za-z_]{20,}|github_pat_[0-9A-Za-z_]{20,}")


def approved_files(root=ROOT):
    root = Path(root)
    return sorted({root / name for name in FILES})


def public_content(path):
    """Normalize release bytes while preserving the local notebook originals."""
    content = path.read_bytes()
    if path.suffix == ".ipynb":
        notebook = json.loads(content)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                cell["outputs"] = []
                cell["execution_count"] = None
        content = (json.dumps(notebook, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
    if path.suffix in {".py", ".md", ".json", ".txt", ".toml", ".yaml", ".ipynb"}:
        if SECRET.search(content.decode("utf-8")):
            raise ValueError("A possible private credential was found; publication stopped.")
    return content


def build_release(root=ROOT):
    output = root / ".runtime/public_release"
    output.mkdir(parents=True, exist_ok=True)
    destination = Path(tempfile.mkdtemp(prefix="harvestguard-", dir=output))
    manifest = []
    for path in approved_files(root):
        relative = path.relative_to(root)
        if (FORBIDDEN_PARTS.intersection(relative.parts) or path.name == "secrets.toml"
                or path.name.startswith(".env") or "profile_quantity_predictions" in path.name
                or path.name.startswith("postharvest_regression_model_ready")):
            raise ValueError("A private or raw-data file entered the public allow-list.")
        content = public_content(path)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        manifest.append({"path": relative.as_posix(), "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)})
    (destination / "PUBLIC_RELEASE_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    archive_path = destination.with_suffix(".zip")
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in destination.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(destination).as_posix())
    result = {"directory": str(destination.resolve()), "archive": str(archive_path.resolve()), "files": len(manifest), "bytes": archive_path.stat().st_size}
    (output / "latest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(build_release(), indent=2))
