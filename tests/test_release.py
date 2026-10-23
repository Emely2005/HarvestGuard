"""Public packaging must include the runnable app and exclude private inputs."""
from pathlib import Path

from scripts.build_public_release import ROOT, FORBIDDEN_PARTS, SECRET, approved_files


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
    assert not SECRET.search("YOUR_GEMINI_API_KEY")
