"""Evidence-constrained decision briefs, separate from model prediction.

This module never scores a model or calls a language model. It preserves saved
predictions and retrieves crop-matched options from a small, cited knowledge base.
"""

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


def load_knowledge_base(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        kb = json.load(handle)
    if kb.get("schema_version") != 1:
        raise ValueError("Unsupported intervention knowledge-base schema.")
    sources = kb.get("sources", {})
    if not sources or not isinstance(kb.get("interventions"), list):
        raise ValueError("The knowledge base needs sources and an intervention list.")
    for source in sources.values():
        for field in ("title", "url", "evidence_type", "scope", "reviewed_on"):
            if not source.get(field):
                raise ValueError(f"An evidence source is missing {field}.")
        if not source["url"].startswith("https://"):
            raise ValueError("Evidence sources must have an HTTPS URL.")
    seen = set()
    for entry in kb["interventions"]:
        for field in (
            "id", "crops", "stage", "title", "action", "when_to_consider",
            "check_first", "evidence_summary", "source_id", "sequence",
        ):
            if not entry.get(field):
                raise ValueError(f"An intervention is missing {field}.")
        if entry["id"] in seen or entry["source_id"] not in sources:
            raise ValueError("Duplicate intervention ID or missing evidence source.")
        if entry["stage"] not in {"storage", "processing", "packaging"}:
            raise ValueError("Interventions must address an in-scope loss stage.")
        if not isinstance(entry["crops"], list) or not all(
            isinstance(crop, str) and crop.strip() for crop in entry["crops"]
        ):
            raise ValueError("Interventions must name their supported crops.")
        if type(entry["sequence"]) is not int or entry["sequence"] < 1:
            raise ValueError("Intervention assessment sequence must be a positive integer.")
        seen.add(entry["id"])
    return kb


def _number(value: Any, name: str, minimum: float, maximum: float) -> float:
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"{name} must be finite and between {minimum} and {maximum}.")
    return number


def build_brief(
    row: Mapping[str, Any],
    metadata: Mapping[str, Any],
    knowledge_base: Mapping[str, Any],
    *,
    district_count: int,
    season_positive_cases: int | None = None,
) -> dict:
    """Use saved facts and crop evidence; abstain on unsupported explanations."""
    crop, district, season = (str(row[key]) for key in ("crop_name", "district", "season"))
    expected = _number(row["predicted_loss_pct"], "Expected loss", 0, 100)
    observed = _number(row["observed_loss_pct"], "Observed loss", 0, 100)
    profiles = _number(row["n_profiles"], "Profile count", 1, math.inf)
    rank = _number(row["display_rank"], "Rank", 1, district_count)
    if not profiles.is_integer() or not rank.is_integer():
        raise ValueError("Profile count and rank must be whole numbers.")
    quantity = "predicted_loss_kg" in row
    scope = str(row.get("loss_quantity_scope", "unavailable"))
    production = expected_kg = observed_kg = None
    if quantity:
        production = _number(row["production_kg"], "Production kg", 0, math.inf)
        expected_kg = _number(row["predicted_loss_kg"], "Expected kg lost", 0, production)
        observed_kg = _number(row["observed_loss_kg"], "Observed kg lost", 0, production)
        if production <= 0 or scope not in {"sample_total", "survey_expanded"}:
            raise ValueError("Quantity briefs require positive production and a documented quantity scope.")
        if not math.isclose(100 * expected_kg / production, expected, abs_tol=1e-6) or not math.isclose(100 * observed_kg / production, observed, abs_tol=1e-6):
            raise ValueError("Quantity totals must match the production-weighted percentages.")

    options = []
    for entry in sorted(knowledge_base["interventions"], key=lambda item: item["sequence"]):
        if crop in entry["crops"]:
            option = deepcopy(entry)
            option["source"] = deepcopy(knowledge_base["sources"][entry["source_id"]])
            options.append(option)

    limitations = [
        "This is a historical 2025 validation view, not a live forecast.",
        "Sample support describes profile counts; it is not a confidence interval.",
        "Local model contributions and district weather values are not in the scoring export; "
        "the brief cannot establish why this district received this prediction.",
        "The combined loss target does not identify which of storage, processing or packaging caused the loss.",
        "Intervention options are matched by crop and ordered for assessment, not by predicted benefit. "
        "Local suitability, cost and effectiveness require field review.",
    ]
    if metadata.get("model_ready_for_national_allocation") is not True:
        limitations.append(
            f"{metadata.get('model_label', 'The model')} has not passed the national allocation readiness checks; use this brief for exploratory review."
        )
    if metadata.get("new_uninspected_period_required") is True:
        limitations.append(
            "The 2025 validation period was previously inspected; a new uninspected period is required before confirming generalisation."
        )
    if not quantity:
        limitations.append("Compatible production quantities are unavailable in the scoring export: kg-based intervention priorities and district tonnes lost cannot be estimated.")
    elif scope == "sample_total":
        limitations.append("Kg totals cover sampled profiles only. They are not district population totals; national prioritisation requires compatible survey expansion weights or complete population coverage.")
    else:
        limitations.append("Kg totals are survey-expanded estimates for the covered profiles and crops; verify survey coverage and weight definitions before national allocation.")
    if not quantity and metadata.get("survey_weights_available") is not True:
        limitations.append("Survey weights are unavailable: these district estimates are not population-weighted totals.")
    if season_positive_cases == 0:
        limitations.append(
            f"The 2025 Season {season} validation export reports zero positive-loss cases; "
            "its ability to rank districts with losses was not established."
        )

    if options:
        next_step = (
            "Ask the district agronomist and cooperative to verify the relevant loss stage and "
            "the prerequisites below before choosing an intervention for a small monitored pilot."
        )
    else:
        next_step = (
            f"No sourced intervention for {crop} is in the current knowledge base. "
            "Request crop-specific evidence and agronomist review before recommending an intervention."
        )

    return {
        "schema_version": 1,
        "mode": "Deterministic evidence brief; see AI advisor for generated explanations",
        "knowledge_base_version": knowledge_base["version"],
        "model": metadata.get("champion", "Unspecified model"),
        "model_label": metadata.get("model_label", "Saved model"),
        "model_version": metadata.get("model_version", metadata.get("candidate_version", "unspecified")),
        "prediction_file": metadata.get("prediction_file", "unspecified"),
        "year": 2025,
        "season": season,
        "crop": crop,
        "district": district,
        "expected_loss_pct": expected,
        "observed_loss_pct": observed,
        "ranking_basis": "predicted_loss_kg" if quantity else "predicted_loss_pct",
        "loss_quantity_scope": scope,
        "production_kg": production,
        "expected_loss_kg": expected_kg,
        "observed_loss_kg": observed_kg,
        "rank": int(rank),
        "districts_with_predictions": district_count,
        "n_profiles": int(profiles),
        "sample_support": str(row["support_level"]),
        "interpretation": (
            f"For {crop} in {district}, Season {season} 2025, estimated loss across storage, processing and packaging is "
            f"{expected_kg:,.2f} kg ({expected_kg / 1000:,.3f} tonnes), or {expected:.2f}% of {production:,.2f} kg production. "
            f"It ranks #{int(rank)} among {district_count} districts by estimated kg lost. "
            + ("These quantities cover sampled profiles only." if scope == "sample_total" else "These quantities are survey-expanded estimates for the covered profiles and crops.")
            if quantity else
            f"For {crop} in {district}, Season {season} 2025, the saved model output estimates "
            f"{expected:.2f}% of crop production lost across storage, processing and packaging. "
            f"It ranks #{int(rank)} among {district_count} districts by loss rate only. Kg-based intervention priorities are unavailable. "
            "This percentage is an expected loss rate, not the probability that a loss occurs."
        ),
        "local_explanation": "Unavailable: district-specific model contributions have not been exported.",
        "weather_context": "Unavailable: the scoring table contains no district weather measurements or forecasts.",
        "hurdle_components": "Occurrence probability and conditional severity are not exported separately.",
        "next_step": next_step,
        "interventions": options,
        "limitations": limitations,
    }


def brief_to_markdown(brief: Mapping[str, Any]) -> str:
    """Export the same evidence and limitations that appear in the interface."""
    if "planning_facts" in brief:
        lines = [f"# HarvestGuard intervention plan: {brief['district']}", "",
                 f"{brief['crop']} | Season {brief['season']} {brief['year']}", "",
                 "Source: saved model evidence", "", brief["interpretation"], "",
                 "## Where to intervene first", "", brief["planning_facts"][0]["text"], "",
                 "## Recommended actions", ""]
        for option in brief["interventions"]:
            lines.extend([f"### {option['title']}", "", option["action"], "", option["rationale"], ""])
            if option["source"].get("url"):
                lines.extend([f"[{option['source']['title']}]({option['source']['url']})", ""])
        lines.extend(["## Data and methods", ""])
        lines.extend(f"- {item}" for item in brief["limitations"])
        return "\n".join(lines) + "\n"
    lines = [
        f"# HarvestGuard Rwanda: {brief['district']} decision brief",
        "",
        f"{brief['crop']} | Season {brief['season']} {brief['year']} | {brief['model']}",
        f"Model: {brief.get('model_label', 'Saved model')} | Version: {brief.get('model_version', 'unspecified')}",
        "",
        f"Mode: {brief['mode']}. Knowledge base: {brief['knowledge_base_version']}.",
        "",
        brief["interpretation"],
        "",
        f"Ranking basis: {brief.get('ranking_basis', 'predicted_loss_pct')}. Quantity scope: {brief.get('loss_quantity_scope', 'unavailable')}.",
        "",
        f"Profiles: {brief['n_profiles']:,}. Sample support: {brief['sample_support']}.",
        f"Observed historical loss: {brief['observed_loss_pct']:.2f}% (validation only).",
        "",
        f"Local explanation: {brief['local_explanation']}",
        f"Weather context: {brief['weather_context']}",
        brief["hurdle_components"],
        "",
        "## Next decision",
        "",
        brief["next_step"],
        "",
        "## Evidence-based options to assess",
        "",
        "The sequence is an assessment workflow, not an estimated effectiveness ranking.",
        "",
    ]
    if not brief["interventions"]:
        lines.extend(["No crop-matched intervention evidence is available.", ""])
    for index, option in enumerate(brief["interventions"], start=1):
        source = option["source"]
        lines.extend([
            f"### {index}. {option['title']} ({option['stage']})",
            "",
            option["action"],
            f"Consider when: {option['when_to_consider']}",
            f"Check first: {option['check_first']}",
            f"Evidence: {option['evidence_summary']}",
            f"Source: [{source['title']}]({source['url']})",
            f"Evidence type: {source['evidence_type']}. Applicability: {source['scope']}",
            f"Source checked: {source['reviewed_on']}. Rwanda agronomic review pending.",
            "",
        ])
    lines.extend(["## Limits of this brief", ""])
    lines.extend(f"- {limitation}" for limitation in brief["limitations"])
    return "\n".join(lines) + "\n"
