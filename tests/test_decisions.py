import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from harvestguard_decisions import build_brief, brief_to_markdown, load_knowledge_base


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def kb():
    return load_knowledge_base(ROOT / "knowledge_base" / "interventions.json")


@pytest.fixture
def row():
    return {
        "district": "Nyagatare", "crop_name": "Maize", "season": "A",
        "predicted_loss_pct": 0.579278, "observed_loss_pct": 0.412245,
        "n_profiles": 411, "display_rank": 1, "support_level": "Strong",
    }


def test_brief_preserves_prediction_units_and_abstains_on_local_explanation(row, kb):
    brief = build_brief(row, {}, kb, district_count=30)
    assert brief["expected_loss_pct"] == row["predicted_loss_pct"]
    assert "0.58%" in brief["interpretation"]
    assert "not the probability" in brief["interpretation"]
    assert brief["local_explanation"].startswith("Unavailable")
    assert brief["weather_context"].startswith("Unavailable")
    assert "confidence interval" in " ".join(brief["limitations"])
    assert "national allocation" in " ".join(brief["limitations"])


def test_crop_specific_evidence_and_storage_prerequisites(row, kb):
    maize = build_brief(row, {}, kb, district_count=30)
    assert [option["id"] for option in maize["interventions"]][:2] == [
        "maize_moisture_before_storage", "maize_hermetic_storage",
    ]
    assert "do not seal wet grain" in maize["interventions"][1]["check_first"]
    tomato = build_brief({**row, "crop_name": "Tomato"}, {}, kb, district_count=30)
    assert [option["id"] for option in tomato["interventions"]] == ["tomato_protective_packaging"]
    unknown = build_brief({**row, "crop_name": "Cassava"}, {}, kb, district_count=30)
    assert unknown["interventions"] == []
    assert "No sourced intervention for Cassava" in unknown["next_step"]


def test_exports_keep_sources_scope_and_cautions(row, kb):
    brief = build_brief(row, {}, kb, district_count=30, season_positive_cases=0)
    markdown = brief_to_markdown(brief)
    for option in brief["interventions"]:
        assert option["source"]["url"] in markdown
        assert option["check_first"] in markdown
        assert option["source"]["scope"] in markdown
    assert "zero positive-loss cases" in markdown
    assert "Rwanda agronomic review pending" in markdown
    assert "Deterministic evidence brief" in markdown
    json.dumps(brief, allow_nan=False)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 101])
def test_invalid_expected_percentages_are_rejected(row, kb, value):
    with pytest.raises(ValueError):
        build_brief({**row, "predicted_loss_pct": value}, {}, kb, district_count=30)


def test_uncited_interventions_fail_closed(kb, tmp_path):
    kb["interventions"][0]["source_id"] = "missing_source"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(kb), encoding="utf-8")
    with pytest.raises(ValueError, match="missing evidence source"):
        load_knowledge_base(path)


def test_all_real_scoring_rows_produce_consistent_briefs(kb):
    metadata = json.loads((ROOT / "tests" / "fixtures" / "legacy_v2" / "model_v2_metadata.json").read_text(encoding="utf-8"))
    data = pd.read_csv(ROOT / "tests" / "fixtures" / "legacy_v2" / "harvestguard_2025_v2_vulnerability.csv")
    original_kb = deepcopy(kb)
    assert not data.duplicated(["season", "crop_name", "district"]).any()
    for (_, crop), group in data.groupby(["season", "crop_name"]):
        group = group.copy()
        group["display_rank"] = group["predicted_loss_pct"].rank(method="first", ascending=False).astype(int)
        for record in group.to_dict("records"):
            brief = build_brief(record, metadata, kb, district_count=len(group))
            assert brief["expected_loss_pct"] == record["predicted_loss_pct"]
            assert brief["n_profiles"] == record["n_profiles"]
            assert brief["districts_with_predictions"] == len(group)
            assert all(crop in option["crops"] for option in brief["interventions"])
            json.dumps(brief, allow_nan=False)
    assert kb == original_kb
