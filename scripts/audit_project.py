"""Check submission files and distinguish deployment needs from provenance gaps."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path

if __package__:
    from .build_public_release import ROOT, approved_files, public_content
else:
    from build_public_release import ROOT, approved_files, public_content


STAGES = {
    "cleaning and unification": [
        "source_preprocessing/build_postharvest_model_ready_corrected.py",
        "source_preprocessing/postharvest_requirements.txt",
        "source_preprocessing/quality_metrics.json",
        "source_preprocessing/postharvest_regression_model_manifest.json",
    ],
    "model development": ["HarvestGuard_Model_Development_V3_Colab.ipynb", "harvestguard_model_v3.py"],
    "saved model": ["model_v3_candidate/model_v3_candidate.joblib", "model_v3_candidate/harvestguard_model_v3.py",
                    "model_v3_candidate/model_v3_metadata.json", "model_v3_candidate/requirements_model_v3.txt"],
    "quantity aggregation": ["HarvestGuard_Quantity_Aggregation_Colab.ipynb",
                             "model_v3_quantity_results/harvestguard_2025_v3_quantity_priorities.csv",
                             "model_v3_quantity_results/quantity_aggregation_metadata.json"],
    "insight export": ["harvestguard_insights_export.py", "model_v3_insights/district_evidence.json"],
    "deployment": ["harvestguard_app.py", "requirements.txt", "requirements-web.txt", ".streamlit/config.toml"],
}

# These are suggested locations, not claims about the original filenames.
SOURCE_GAPS = {
    "2025 raw production/practice merge and season combination": "source_preprocessing/merge_2025_seasons.py",
    "exact production-bearing export and 10-row reconciliation": "source_preprocessing/build_production_input.py",
}
PRIVATE_INPUTS = ["postharvest_regression_model_ready_2024_2025.csv",
                  "postharvest_regression_model_ready_2024_2025_with_production.csv"]
RAW_INPUTS = ["Dataset.zip", "all_seasons.zip", "Climate.zip"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root=ROOT, data_dir=None, raw_dir=None, verify_manifest=False):
    root = Path(root).resolve()
    errors = []
    stages = {stage: {name: (root / name).is_file() for name in names} for stage, names in STAGES.items()}
    files = []
    for path in approved_files(root):
        relative = path.relative_to(root).as_posix()
        if not path.is_file():
            errors.append("Missing submission file: " + relative)
            continue
        try:
            content = public_content(path)
        except ValueError as error:
            errors.append(f"Invalid public content: {relative}: {error}")
            continue
        files.append({"path": relative, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
        try:
            if path.suffix == ".py":
                ast.parse(content.decode("utf-8-sig"), filename=relative)
            elif path.suffix in {".json", ".ipynb"}:
                json.loads(content)
        except (SyntaxError, ValueError) as error:
            errors.append(f"Invalid source or JSON: {relative}: {error}")
    for stage, paths in stages.items():
        for name, exists in paths.items():
            if not exists:
                errors.append(f"Missing {stage} artifact: {name}")
    try:
        quantity = json.loads((root / "model_v3_quantity_results/quantity_aggregation_metadata.json").read_text(encoding="utf-8"))
        evidence = json.loads((root / "model_v3_insights/district_evidence.json").read_text(encoding="utf-8"))
        checks = {
            "saved model matches quantity export": digest(root / "model_v3_candidate/model_v3_candidate.joblib") == quantity["source_model_sha256"],
            "insights match saved model": evidence["source_model_sha256"] == quantity["source_model_sha256"],
            "insights match quantity CSV": digest(root / "model_v3_quantity_results/harvestguard_2025_v3_quantity_priorities.csv") == evidence["quantity_csv_sha256"],
            "insights match importance CSV": digest(root / "model_v3_insights/feature_importance_hurdle.csv") == evidence["importance_csv_sha256"],
            "insights and quantities use same input": evidence["input_sha256"] == quantity["input_sha256"],
        }
        errors.extend(name for name, passed in checks.items() if not passed)
    except (OSError, KeyError, ValueError) as error:
        errors.append("Cannot validate model/export provenance: " + str(error))
        checks = {}
        quantity = {}
    gaps = [{"step": step, "suggested_path": name} for step, name in SOURCE_GAPS.items() if not (root / name).is_file()]
    private = {}
    if data_dir is not None:
        for name in PRIVATE_INPUTS:
            path = Path(data_dir) / name
            record = {"present": path.is_file()}
            if path.is_file():
                with path.open(encoding="utf-8-sig", newline="") as handle:
                    reader = csv.DictReader(handle)
                    record["columns"] = len(reader.fieldnames)
                    years = {}
                    for row in reader:
                        years[row["year"]] = years.get(row["year"], 0) + 1
                    record.update(rows=sum(years.values()), rows_by_year=years, sha256=digest(path))
                if name == quantity.get("input_file") and record["sha256"] != quantity["input_sha256"]:
                    errors.append("Private scoring input differs from the recorded export input.")
            private[name] = record
    raw = {name: (Path(raw_dir) / name).is_file() for name in RAW_INPUTS} if raw_dir is not None else {}
    if verify_manifest:
        try:
            recorded = json.loads((root / "PUBLIC_RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
            if sorted(recorded, key=lambda row: row["path"]) != sorted(files, key=lambda row: row["path"]):
                errors.append("PUBLIC_RELEASE_MANIFEST.json differs from the approved submission files; rebuild the release.")
        except (OSError, ValueError) as error:
            errors.append("Cannot read release manifest: " + str(error))
    return {"submission_valid": not errors, "errors": errors, "stages": stages, "provenance_checks": checks,
            "unresolved_source_steps": gaps, "private_inputs": private, "raw_inputs": raw,
            "public_file_count": len(files), "inventory": files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, help="Private model-ready CSV directory; never packaged.")
    parser.add_argument("--raw-dir", type=Path, help="Directory containing the three original input archives.")
    parser.add_argument("--verify-manifest", action="store_true")
    parser.add_argument("--require-reproducibility", action="store_true", help="Fail for unresolved source steps or missing checked private/raw inputs.")
    parser.add_argument("--json", type=Path, help="Save complete machine-readable report.")
    args = parser.parse_args()
    report = audit(data_dir=args.data_dir, raw_dir=args.raw_dir, verify_manifest=args.verify_manifest)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Submission: {'PASS' if report['submission_valid'] else 'FAIL'}; {report['public_file_count']} approved files")
    for stage, paths in report["stages"].items():
        print(f"{'PRESENT' if all(paths.values()) else 'MISSING'}: {stage}")
    for error in report["errors"]:
        print("ERROR: " + error)
    for gap in report["unresolved_source_steps"]:
        print("MISSING SOURCE STEP: " + gap["step"] + " (suggested location: " + gap["suggested_path"] + ")")
    for name, record in report["private_inputs"].items():
        print(f"Private input {name}: " + (f"{record['rows']:,} rows" if record["present"] else "MISSING"))
    for name, present in report["raw_inputs"].items():
        print(f"Raw input {name}: {'PRESENT' if present else 'MISSING'}")
    incomplete = bool(report["unresolved_source_steps"]) or any(not r["present"] for r in report["private_inputs"].values()) or any(not p for p in report["raw_inputs"].values())
    return 1 if report["errors"] else 2 if args.require_reproducibility and incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
