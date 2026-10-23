"""Auditable intervention planning from verified aggregate model evidence."""

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

import pandas as pd


GRAIN_CROPS = {"Maize", "Paddy rice", "Sorghum", "Bush bean", "Climbing bean", "Pea", "Soybean", "Groundnut", "Sunflower"}
WEATHER_SOURCE = {
    "title": "FAO: The fundamentals of storage",
    "url": "https://www.fao.org/4/x5065e/x5065e04.htm",
    "scope": "Moisture control in stored cereals and legumes; weather does not measure grain moisture.",
}
MODEL_SOURCE = {"title": "HarvestGuard aggregate profile analysis", "url": "", "scope": "Saved 2025 predictions; loss quantities calculated before aggregation."}


@lru_cache(maxsize=4)
def _read_export(path_string, stamp, model_hash, quantity_hash):
    data = json.loads(Path(path_string).read_text(encoding="utf-8"))
    if (data.get("schema_version") != 1 or data.get("model_retrained") is not False
            or data.get("source_model_sha256") != model_hash or data.get("quantity_csv_sha256") != quantity_hash):
        raise ValueError("District evidence does not match the displayed model and quantity results.")
    records = {}
    for record in data["records"]:
        key = tuple(record[name] for name in ("district", "crop", "season", "year"))
        if key in records:
            raise ValueError("District evidence contains duplicate groups.")
        records[key] = record
    return data, records


def load_insight_evidence(app_dir, model_dir, brief):
    path = app_dir / "model_v3_insights" / "district_evidence.json"
    if not path.exists() or brief.get("ranking_basis") != "predicted_loss_kg":
        return None, None, []
    quantity_path = app_dir / brief["prediction_file"]
    model_hash = hashlib.sha256((model_dir / "model_v3_candidate.joblib").read_bytes()).hexdigest()
    quantity_hash = hashlib.sha256(quantity_path.read_bytes()).hexdigest()
    stat = path.stat()
    data, records = _read_export(str(path), (stat.st_mtime_ns, stat.st_size), model_hash, quantity_hash)
    record = records.get(tuple(brief[name] for name in ("district", "crop", "season", "year")))
    if record is None:
        return None, None, []
    for actual, expected in ((record["expected_loss_pct"], brief["expected_loss_pct"]),
                             (record["expected_loss_kg"], brief["expected_loss_kg"]),
                             (record["production_kg"], brief["production_kg"])):
        if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-6):
            raise ValueError("District evidence belongs to a different saved prediction.")
    if record["n_profiles"] != brief["n_profiles"]:
        raise ValueError("District evidence profile support differs from the prediction.")
    importance_path = path.parent / "feature_importance_hurdle.csv"
    # The CSV is a convenient display only. Its provenance and values are bound
    # to the evidence JSON rather than trusted as an independent loose file.
    if hashlib.sha256(importance_path.read_bytes()).hexdigest() != data.get("importance_csv_sha256"):
        raise ValueError("Global importance does not match its evidence export.")
    importance = pd.read_csv(importance_path).to_dict("records")
    info = {key: value for key, value in data.items() if key != "records"}
    # Retrieve a small, relevant slice from the hash-verified aggregate export.
    # Farmer rows and unrelated project files never enter the AI request.
    info["comparative_records"] = [
        {key: item[key] for key in ("district", "crop", "season", "year", "expected_loss_pct", "expected_loss_kg", "production_kg", "n_profiles")}
        for item in records.values()
        if item["district"] == brief["district"] and item["year"] == brief["year"] and item["n_profiles"] >= 10
        and (item["season"] == brief["season"] or item["crop"] == brief["crop"])
    ]
    return deepcopy(record), info, importance


def plan_interventions(brief, peers, record=None, *, comparative_records=None):
    """Select actions by explicit evidence rules, never just by the crop name."""
    brief = deepcopy(brief)
    quantity = brief.get("ranking_basis") == "predicted_loss_kg"
    if not quantity:
        return brief
    peers = sorted(peers, key=lambda row: (-float(row["predicted_loss_kg"]), row["district"]))
    total = sum(float(row["predicted_loss_kg"]) for row in peers)
    leaders = peers[:3]
    share = 100 * sum(float(row["predicted_loss_kg"]) for row in leaders) / total if total else 0.0
    display = {"priority_share": f"{share:.1f}%", "selection_share": f"{100 * brief['expected_loss_kg'] / total:.1f}%" if total else "0.0%"}
    facts = [{"id": "priority_plan", "kind": "comparison",
              "text": "Intervene first in " + ", ".join(f"{row['district']} ({float(row['predicted_loss_kg']):,.2f} kg)" for row in leaders)
              + f". Together these districts account for {share:.1f}% of estimated loss in this crop-season selection. Ranking uses sampled kg totals, not treatment benefits.",
              "source_title": MODEL_SOURCE["title"]}]
    options = []

    def add_option(identifier, title, action, finding, refs, stage="storage", source=None):
        options.append({"id": identifier, "title": title, "action": action, "rationale": finding,
                        "required_evidence_ids": refs, "stage": stage, "sequence": len(options) + 1,
                        "evidence_summary": finding, "source": source or MODEL_SOURCE,
                        "when_to_consider": "Selected from the matching district evidence.",
                        "check_first": "", "crops": [brief["crop"]]})

    if record:
        for feature, weather in record.get("weather", {}).items():
            short = "humidity" if "humidity" in feature else "rainfall"
            suffix = "%" if short == "humidity" else " mm"
            if weather["value"] is not None:
                display[short] = f"{weather['value']:.1f}{suffix}"
                display[short + "_median"] = f"{weather['peer_median']:.1f}{suffix}"
                facts.append({"id": "weather_" + short, "kind": "measured_context",
                    "text": f"Production-weighted {short} before harvest: {display[short]}; median across same-crop/season district means: {display[short + '_median']}. Coverage: {weather['coverage_pct']:.1f}% of production. This is ambient weather, not measured storage or grain moisture.",
                    "source_title": MODEL_SOURCE["title"]})
        # Only link weather services when observed exposure AND a positive local
        # full-model sensitivity support that link. No equipment is inferred.
        for feature in ("humidity_mean_7d_before_pct", "rain_30d_before_mm"):
            weather = record.get("weather", {}).get(feature)
            driver = next((item for item in record.get("drivers", []) if item["feature"] == feature), None)
            if (brief["crop"] in GRAIN_CROPS and weather and driver and weather["value"] is not None
                    and weather["coverage_pct"] >= 90 and weather["value"] > weather["peer_median"]
                    and driver["contribution"] > 0 and driver in record["drivers"][:3]):
                short = "humidity" if "humidity" in feature else "rainfall"
                driver_index = record["drivers"].index(driver)
                title = "Allocate drying and moisture-control services" if short == "humidity" else "Provide rain-protected drying capacity"
                action = f"Direct covered drying space and moisture-testing services to {brief['district']}'s {brief['crop'].lower()} programme; keep dried grain protected from rewetting."
                rationale = f"{short.capitalize()} is {display[short]}, above the district comparison median of {display[short + '_median']}, and it raises the model estimate against the development reference."
                add_option("weather_moisture_services", title, action, rationale,
                           ["weather_" + short, f"local_model_{driver_index}"], source=WEATHER_SOURCE)
                break
        concentration = record.get("concentration")
        if concentration and concentration["loss_share_pct"] > concentration["profile_share_pct"]:
            display.update(cohort_profile_share=f"{concentration['profile_share_pct']:.1f}%", cohort_loss_share=f"{concentration['loss_share_pct']:.1f}%",
                           cohort_loss_kg=f"{concentration['loss_kg']:,.2f} kg")
            finding = f"The largest-production {display['cohort_profile_share']} of sampled profiles account for {display['cohort_loss_share']} of estimated loss ({display['cohort_loss_kg']})."
            facts.append({"id": "loss_concentration", "kind": "profile_summary", "text": finding, "source_title": MODEL_SOURCE["title"]})
            add_option("production_capacity", "Direct capacity to the largest loss volumes",
                       f"Prioritize high-volume production batches in {brief['district']}'s {brief['crop'].lower()} processing, storage and packaging programme.",
                       finding, ["loss_concentration"], stage="processing")
        timing = record.get("timing")
        if timing:
            display.update(peak_month=timing["month"], peak_loss_share=f"{timing['loss_share_pct']:.1f}%", peak_loss_kg=f"{timing['loss_kg']:,.2f} kg")
            finding = f"Profiles harvested in {timing['month']} account for {display['peak_loss_share']} of this group's estimated postharvest loss ({display['peak_loss_kg']}). This is a harvest-cohort service window, not an observed date of loss."
            facts.append({"id": "service_timing", "kind": "profile_summary", "text": finding, "source_title": MODEL_SOURCE["title"]})
            add_option("seasonal_services", f"Schedule services around {timing['month']}",
                       f"Align {brief['district']}'s processing, storage and packaging support with the {timing['month']} harvest cohort in this seasonal plan.",
                       f"The {timing['month']} cohort accounts for {display['peak_loss_share']} of estimated loss ({display['peak_loss_kg']}).", ["service_timing"], stage="processing")
    if not options:
        add_option("district_allocation", "Allocate support by loss volume",
                   f"Place {brief['district']} at priority {brief['rank']} of {brief['districts_with_predictions']} in the {brief['crop'].lower()} seasonal plan, using its estimated {brief['expected_loss_kg']:,.2f} kg loss.",
                   "The allocation order comes directly from summed profile losses.", ["prediction", "ranking"])
    findings = []
    if comparative_records is not None:
        rates = [float(row["predicted_loss_pct"]) for row in peers]
        rate_rank = 1 + sum(value > brief["expected_loss_pct"] for value in rates)
        median_rate = float(pd.Series(rates).median())
        display.update(rate_rank=str(rate_rank), peer_rate_median=f"{median_rate:.2f}%")
        comparison = (f"{brief['district']} is priority {brief['rank']} by estimated loss kg but rank {rate_rank} by production-weighted loss percentage. "
                      f"Its rate is {brief['expected_loss_pct']:.2f}%; the same-crop/season district median is {median_rate:.2f}%. "
                      "Loss volume identifies where more sampled kg are lost; loss percentage identifies handling-loss intensity.")
        facts.append({"id": "volume_rate_comparison", "kind": "profile_summary", "text": comparison, "source_title": MODEL_SOURCE["title"]})
        if abs(brief["rank"] - rate_rank) >= 3:
            findings.append(("Compare the indicators", "Loss intensity and volume tell different stories", comparison))
        if brief["n_profiles"] >= 10 and rate_rank <= max(3, math.ceil(len(peers) / 5)) and brief["rank"] > rate_rank + 3 and brief["expected_loss_pct"] > median_rate:
            add_option("targeted_loss_control", "Target the high loss rate",
                       f"Assign handling-loss inspections to {brief['district']}'s {brief['crop'].lower()} batches. Record storage, processing and packaging losses separately to direct follow-up.",
                       comparison, ["volume_rate_comparison", "prediction"], stage="processing")

        portfolio = [item for item in comparative_records if item["season"] == brief["season"]]
        if len(portfolio) >= 2:
            total_loss = sum(item["expected_loss_kg"] for item in portfolio)
            selected = next((item for item in portfolio if item["crop"] == brief["crop"]), None)
            leader = max(portfolio, key=lambda item: item["expected_loss_kg"])
            if selected and total_loss > 0:
                share = 100 * selected["expected_loss_kg"] / total_loss
                display.update(district_crop_share=f"{share:.1f}%", district_leading_crop=leader["crop"],
                               district_leading_crop_share=f"{100 * leader['expected_loss_kg'] / total_loss:.1f}%")
                finding = (f"Within {brief['district']}'s supported crop groups in Season {brief['season']}, {brief['crop']} accounts for {share:.1f}% of estimated sampled loss kg. "
                           f"{leader['crop']} has the largest estimated volume, with {display['district_leading_crop_share']}. This comparison covers supported crop groups, not all district agriculture.")
                facts.append({"id": "district_crop_portfolio", "kind": "profile_summary", "text": finding, "source_title": MODEL_SOURCE["title"]})
                findings.append(("Across the district", "Where this crop fits in the district programme", finding))
                if leader["crop"] == brief["crop"] and share >= 50:
                    add_option("crop_programme_focus", f"Focus the district programme on {brief['crop'].lower()}",
                               f"Give {brief['crop'].lower()} the largest share of {brief['district']}'s Season {brief['season']} handling-loss support among the covered crop programmes.",
                               finding, ["district_crop_portfolio"], stage="processing")

        seasons = [item for item in comparative_records if item["crop"] == brief["crop"]]
        if len(seasons) >= 2:
            total_loss = sum(item["expected_loss_kg"] for item in seasons)
            peak = max(seasons, key=lambda item: item["expected_loss_kg"])
            if total_loss > 0:
                peak_share = 100 * peak["expected_loss_kg"] / total_loss
                display.update(district_peak_season=peak["season"], district_peak_season_share=f"{peak_share:.1f}%")
                finding = (f"For {brief['district']}'s {brief['crop']} across supported 2025 seasons, Season {peak['season']} contains {peak_share:.1f}% of estimated loss kg. "
                           "These are separate historical seasonal cohorts, not a forecast or an estimate of losses avoided.")
                facts.append({"id": "district_season_pattern", "kind": "profile_summary", "text": finding, "source_title": MODEL_SOURCE["title"]})
                findings.append(("Across seasons", f"Season {peak['season']} carries the largest loss volume", finding))
                if peak["season"] == brief["season"] and peak_share >= 60:
                    add_option("season_programme_focus", f"Concentrate the seasonal plan in Season {peak['season']}",
                               f"Prioritize Season {peak['season']} in {brief['district']}'s {brief['crop'].lower()} handling-loss programme, matching support to the supplied harvest-cohort service window.",
                               finding, ["district_season_pattern", "service_timing"] if record and record.get("timing") else ["district_season_pattern"], stage="processing")
        priorities = {"weather_moisture_services": 100, "targeted_loss_control": 95, "crop_programme_focus": 88,
                      "season_programme_focus": 85, "production_capacity": 80, "seasonal_services": 75, "district_allocation": 50}
        options.sort(key=lambda option: -priorities[option["id"]])
    brief.update(interventions=options if comparative_records is not None else options[:2], planning_facts=facts, planning_values=display,
                 discovered_findings=findings[:2],
                 priority_districts=[{"district": row["district"], "loss_kg": float(row["predicted_loss_kg"]),
                                      "rank": 1 + sum(float(peer["predicted_loss_kg"]) > float(row["predicted_loss_kg"]) for peer in peers)} for row in leaders],
                 priority_share=share, district_evidence=record)
    if record and record.get("drivers"):
        brief["local_explanation"] = "District numeric sensitivities were recomputed with the unchanged complete calibrated model against 2024 crop-season reference values. These comparisons are not causal effects."
        brief["weather_context"] = "Production-weighted preharvest humidity and rainfall are supplied with district peer comparisons."
        brief["limitations"] = [item for item in brief["limitations"] if not item.startswith(("Local model contributions", "Intervention options"))]
    return brief
