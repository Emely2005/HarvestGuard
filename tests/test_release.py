"""Public packaging must include the runnable app and exclude private inputs."""
from pathlib import Path

from scripts.build_public_release import ROOT, FORBIDDEN_PARTS, SECRET, approved_files
from scripts.audit_project import audit


def test_release_allowlist_excludes_secrets_runtime_and_farmer_predictions():
    files = approved_files()
    assert files
    for path in files:
        assert path.is_file(), path
        relative = path.relative_to(ROOT)
        assert not FORBIDDEN_PARTS.intersection(relative.parts)
        assert path.name != "secrets.toml"
        assert "profile_quantity_predictions" not in path.name
        assert not path.name.startswith("postharvest_regression_model_ready")
    names = {path.relative_to(ROOT).as_posix() for path in files}
    assert {"Dockerfile", "render.yaml", "harvestguard_methodology.py", "METHODOLOGY.md", "ddi-documentation-english-124.pdf"}.issubset(names)


def test_credential_scan_recognises_common_provider_token_formats():
    assert SECRET.search("AIza" + "x" * 35)
    assert SECRET.search("github_pat_" + "x" * 35)
    assert SECRET.search("gsk_" + "x" * 48)
    assert not SECRET.search("YOUR_GEMINI_API_KEY")


def test_release_does_not_pick_up_stray_files_in_retained_directories(tmp_path):
    stray = tmp_path / "source_preprocessing" / "old_cleaning_draft.py"
    stray.parent.mkdir()
    stray.write_text("# superseded experiment", encoding="utf-8")
    assert stray not in approved_files(tmp_path)
    names = {path.relative_to(ROOT).as_posix() for path in approved_files()}
    assert not any(name.startswith("model_v2/") for name in names)
    assert "model_v3_candidate/ranking_detail_candidate.csv" not in names
    assert "tests/fixtures/legacy_v2/model_v2_metadata.json" in names


def test_audit_validates_current_exports_and_reports_missing_source_history():
    report = audit()
    assert report["submission_valid"], report["errors"]
    assert all(report["provenance_checks"].values())
    assert len(report["unresolved_source_steps"]) == 2


def test_audit_reports_missing_workflow_artifacts_and_private_inputs(tmp_path):
    report = audit(root=tmp_path, data_dir=tmp_path, raw_dir=tmp_path)
    assert not report["submission_valid"]
    assert not any(report["stages"]["cleaning and unification"].values())
    assert not any(report["stages"]["model development"].values())
    assert not any(record["present"] for record in report["private_inputs"].values())
    assert not any(report["raw_inputs"].values())
