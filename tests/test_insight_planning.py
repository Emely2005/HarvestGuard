"""Real-export provenance and district-specific recommendation regressions."""
from copy import deepcopy
import json
from pathlib import Path

import pandas as pd
import pytest

from harvestguard_ai import AIResponseError, build_evidence_packet, validate_analysis
from harvestguard_data import load_dashboard_data, rank_districts
from harvestguard_decisions import build_brief, load_knowledge_base
from harvestguard_insights import _read_export, load_insight_evidence, plan_interventions
from test_ai import valid_response

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def evidence():
    risk, metadata, _ = load_dashboard_data(ROOT / "model_v3_candidate")
    kb = load_knowledge_base(ROOT / "knowledge_base" / "interventions.json")
    export = json.loads((ROOT / "model_v3_insights" / "district_evidence.json").read_text(encoding="utf-8"))
    return risk, metadata, kb, export


def make_plan(evidence, district="Nyagatare", crop="Maize", season="A", *, enriched=False):
    risk, metadata, kb, _ = evidence
    view = rank_districts(risk.loc[risk.crop_name.eq(crop) & risk.season.eq(season)].copy())
    row = view.loc[view.district.eq(district)].iloc[0].to_dict()
    brief = build_brief(row, metadata, kb, district_count=len(view))
    record, info, importance = load_insight_evidence(ROOT, ROOT / "model_v3_candidate", brief)
    plan = plan_interventions(brief, view.to_dict("records"), record,
                              comparative_records=info["comparative_records"] if enriched else None)
    local = {**record, "expected_loss_pct": brief["expected_loss_pct"], "method": info["method"], "aggregation": info["aggregation"]} if record["drivers"] else None
    packet = build_evidence_packet(plan, view.to_dict("records"), global_importance=importance, local_explanation=local)
    return plan, packet, info, importance


def test_default_plan_uses_actual_loss_concentration_and_timing(evidence):
    plan, packet, _, importance = make_plan(evidence)
    assert [option["id"] for option in plan["interventions"]] == ["production_capacity", "seasonal_services"]
    assert plan["district_evidence"]["concentration"]["loss_share_pct"] == pytest.approx(98.316851)
    assert plan["district_evidence"]["timing"]["month"] == "January"
    assert packet["display_values"]["peak_loss_share"] == "60.4%"
    assert [entry["feature"] for entry in importance[:3]] == ["crop_name", "crop_plot_area_ha_total", "district"]
    assert "weather_moisture_services" not in [option["id"] for option in plan["interventions"]]
    assert any(fact["kind"] == "local_model" for fact in packet["facts"])
    assert "District-level weather values" not in packet["unavailable"]


def test_priority_stays_on_kg_leaders_when_lower_ranked_district_selected(evidence):
    plan, _, _, _ = make_plan(evidence, district="Nyarugenge")
    assert plan["rank"] == 26
    assert [row["district"] for row in plan["priority_districts"]] == ["Nyagatare", "Kirehe", "Kamonyi"]
    assert plan["priority_districts"][0]["loss_kg"] == pytest.approx(64545.969, abs=.01)
    assert "hermetic" not in json.dumps(plan["interventions"]).lower()


def test_equal_loss_volumes_share_intervention_priority(evidence):
    plan, _, _, _ = make_plan(evidence)
    peers = evidence[0].loc[evidence[0].crop_name.eq("Maize") & evidence[0].season.eq("A")].to_dict("records")
    maximum = max(row["predicted_loss_kg"] for row in peers)
    for row in peers:
        if row["district"] == "Kirehe":
            row["predicted_loss_kg"] = maximum
    tied = plan_interventions(plan, peers, plan["district_evidence"])
    assert [row["rank"] for row in tied["priority_districts"]] == [1, 1, 3]


def test_weather_action_requires_both_measured_exposure_and_local_model_signal(evidence):
    _, _, _, export = evidence
    candidates = []
    for record in export["records"]:
        if record["crop"] != "Maize" or not record["drivers"]:
            continue
        plan, _, _, _ = make_plan(evidence, record["district"], record["crop"], record["season"])
        if any(option["id"] == "weather_moisture_services" for option in plan["interventions"]):
            candidates.append(plan)
            break
    assert candidates, "The real export should contain a supported weather-action example."
    plan = candidates[0]
    option = plan["interventions"][0]
    assert any(ref.startswith("weather_") for ref in option["required_evidence_ids"])
    assert any(ref.startswith("local_model_") for ref in option["required_evidence_ids"])
    record = deepcopy(plan["district_evidence"])
    for weather in record["weather"].values():
        weather["value"] = weather["peer_median"] - 1
    risk = evidence[0]
    peers = risk.loc[risk.crop_name.eq(plan["crop"]) & risk.season.eq(plan["season"])].to_dict("records")
    changed = plan_interventions(plan, peers, record)
    assert all(item["id"] != "weather_moisture_services" for item in changed["interventions"])
    record = deepcopy(plan["district_evidence"])
    for driver in record["drivers"]:
        driver["contribution"] = -abs(driver["contribution"])
    changed = plan_interventions(plan, peers, record)
    assert all(item["id"] != "weather_moisture_services" for item in changed["interventions"])


def test_ai_recommendations_must_cite_the_district_findings(evidence):
    _, packet, _, _ = make_plan(evidence)
    response = valid_response(packet)
    validate_analysis(response, packet)
    option = packet["interventions"][0]
    response["recommendations"][0]["evidence_ids"] = ["intervention_" + option["id"]]
    with pytest.raises(AIResponseError, match="district findings"):
        validate_analysis(response, packet)


def test_profile_insights_cannot_claim_statistical_significance(evidence):
    _, packet, _, _ = make_plan(evidence)
    response = valid_response(packet)
    response["insights"] = [{"type": "context", "text": "These profiles show significant concentration of estimated loss.", "evidence_ids": ["loss_concentration"]}]
    with pytest.raises(AIResponseError, match="overstated"):
        validate_analysis(response, packet)


def test_provenance_rejects_another_model_or_quantity_table(evidence):
    export = evidence[3]
    path = ROOT / "model_v3_insights" / "district_evidence.json"
    with pytest.raises(ValueError, match="does not match"):
        _read_export(str(path), "bad-model", "wrong", export["quantity_csv_sha256"])
    with pytest.raises(ValueError, match="does not match"):
        _read_export(str(path), "bad-quantities", export["source_model_sha256"], "wrong")


def test_small_groups_suppress_detailed_profile_summaries(evidence):
    records = evidence[3]["records"]
    assert len(records) == 944
    assert sum(record["n_profiles"] for record in records) == 50475
    small = [record for record in records if record["n_profiles"] < 10]
    assert small
    assert all(not record["drivers"] and not record["weather"] and "timing" not in record and "concentration" not in record for record in small)


def test_all_crop_plans_use_group_evidence_without_crop_only_equipment(evidence):
    for crop in sorted(evidence[0].crop_name.unique()):
        rows = evidence[0].loc[evidence[0].crop_name.eq(crop) & evidence[0].season.eq("A")]
        row = rows.sort_values("predicted_loss_kg", ascending=False).iloc[0]
        plan, packet, _, _ = make_plan(evidence, row.district, crop)
        assert plan["interventions"]
        for option in plan["interventions"]:
            assert option["required_evidence_ids"]
            assert all(ref in {fact["id"] for fact in packet["facts"]} for ref in option["required_evidence_ids"])
        assert "hermetic" not in json.dumps(plan["interventions"]).lower()


def test_retrieval_uses_only_supported_relevant_district_records(evidence):
    plan, packet, info, _ = make_plan(evidence, enriched=True)
    records = info["comparative_records"]
    assert records
    assert all(row["district"] == plan["district"] and row["year"] == plan["year"] and row["n_profiles"] >= 10 for row in records)
    assert all(row["crop"] == plan["crop"] or row["season"] == plan["season"] for row in records)
    assert {"volume_rate_comparison", "district_crop_portfolio", "district_season_pattern"}.issubset({fact["id"] for fact in packet["facts"]})
    json.dumps(packet, allow_nan=False)


def test_different_districts_get_different_supported_action_choices(evidence):
    large, packet, _, _ = make_plan(evidence, enriched=True)
    intense, other, _, _ = make_plan(evidence, "Nyarugenge", enriched=True)
    assert large["rank"] == 1 and intense["rank"] == 26
    assert intense["planning_values"]["rate_rank"] == "1"
    assert any(option["id"] == "targeted_loss_control" for option in intense["interventions"])
    assert [item["id"] for item in large["interventions"][:2]] != [item["id"] for item in intense["interventions"][:2]]
    for plan, evidence_packet in [(large, packet), (intense, other)]:
        ids = {fact["id"] for fact in evidence_packet["facts"]}
        for option in plan["interventions"]:
            assert set(option["required_evidence_ids"]).issubset(ids)
        validate_analysis(valid_response(evidence_packet), evidence_packet)


def test_portfolio_shares_are_recomputed_from_matching_aggregate_data(evidence):
    plan, _, info, _ = make_plan(evidence, enriched=True)
    selected = [row for row in info["comparative_records"] if row["season"] == plan["season"]]
    numerator = next(row["expected_loss_kg"] for row in selected if row["crop"] == plan["crop"])
    expected = 100 * numerator / sum(row["expected_loss_kg"] for row in selected)
    assert plan["planning_values"]["district_crop_share"] == f"{expected:.1f}%"
