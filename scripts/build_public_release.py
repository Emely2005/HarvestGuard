"""Package a public release from an explicit allow-list; never include raw data."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "harvestguard_app.py", "dashboard.py", "harvestguard_ai.py", "harvestguard_ai_cache.py", "harvestguard_ai_view.py",
    "harvestguard_data.py", "harvestguard_decisions.py", "harvestguard_insights.py", "harvestguard_insights_export.py",
    "harvestguard_map.py", "harvestguard_ui.py", "harvestguard_views.py", "harvestguard_methodology.py", "harvestguard_model_v3.py",
    "README.md", "METHODOLOGY.md", "DEPLOYMENT.md", "SUBMISSION.md", "AI_SETUP.md", "MODEL_REVIEW.md",
    "requirements.txt", "requirements-web.txt", "requirements-dev.txt", "Dockerfile", "render.yaml", ".dockerignore", ".gitignore",
    ".streamlit/config.toml", ".streamlit/secrets.toml.example",
    "ddi-documentation-english-113.pdf", "ddi-documentation-english-124.pdf",
    "HarvestGuard_Model_Development_V3_Colab.ipynb", "HarvestGuard_Quantity_Aggregation_Colab.ipynb",
    "model_v3_quantity_results/harvestguard_2025_v3_quantity_priorities.csv",
    "model_v3_quantity_results/quantity_aggregation_metadata.json", "model_v3_quantity_results/RUN_RESULTS.md",
    "model_v3_insights/district_evidence.json", "model_v3_insights/feature_importance_hurdle.csv", "model_v3_insights/RUN_RESULTS.md",
    "model_v2/model_v2_metadata.json", "model_v2/harvestguard_2025_v2_vulnerability.csv",
    "scripts/build_public_release.py",
]
DIRECTORIES = ["knowledge_base", "model_v3_candidate", "source_preprocessing", "tests", "model_review_tests"]
FORBIDDEN_PARTS = {".git", ".runtime", ".aws", ".codex", ".agents", "__pycache__", ".pytest_cache", ".venv"}
SECRET = re.compile(r"AIza[0-9A-Za-z_-]{35}|sk-[0-9A-Za-z_-]{24,}|ghp_[0-9A-Za-z_]{20,}|github_pat_[0-9A-Za-z_]{20,}")


def approved_files(root=ROOT):
    paths = {root / name for name in FILES}
    for directory in DIRECTORIES:
        paths.update(path for path in (root / directory).rglob("*") if path.is_file()
                     and not FORBIDDEN_PARTS.intersection(path.relative_to(root).parts))
    return sorted(paths)


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
