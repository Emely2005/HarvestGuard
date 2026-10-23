"""Intervention priorities, visible model findings and evidence-linked AI wording."""

import hashlib
import json
import os
from pathlib import Path
import time
import tomllib

import streamlit as st

from harvestguard_ai import (
    DEFAULT_MODEL, build_evidence_packet, generate_analysis, load_context,
    load_local_explanation, render_text, request_fingerprint,
)
from harvestguard_ai_cache import lookup_explanation, request_explanation, retry_explanation
from harvestguard_data import MODEL_FOLDER
from harvestguard_decisions import brief_to_markdown
from harvestguard_insights import load_insight_evidence, plan_interventions
from harvestguard_ui import card, priority_strip

PRIVATE_SETTINGS = Path(__file__).resolve().parent / ".streamlit" / "secrets.toml"


def _setting(name, default=""):
    value = os.environ.get(name, "")
    if value:
        return value
    try:
        private = tomllib.loads(PRIVATE_SETTINGS.read_text(encoding="utf-8"))
        if private.get(name):
            return str(private[name])
    except (OSError, ValueError):
        pass
    try:
        return str(st.secrets.get(name, default))
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return default


def _sync_configured_key(server_key):
    version = hashlib.sha256(server_key.encode("utf-8")).hexdigest() if server_key else None
    if st.session_state.get("_configured_key_version") != version:
        st.session_state["_configured_key_version"] = version
        for name in ("_gemini_key", "_ai_result", "_ai_failure", "_ai_cache", "_ai_retry_cycles"):
            st.session_state.pop(name, None)


def _references(statement, packet):
    lookup = {fact["id"]: fact for fact in packet["facts"]}
    references = []
    for ref in dict.fromkeys(statement["evidence_ids"]):
        fact = lookup[ref]
        references.append(f"[{fact.get('source_title', ref)}]({fact['source_url']})" if fact.get("source_url") else fact.get("source_title", ref.replace("_", " ")))
    return "Evidence: " + " · ".join(dict.fromkeys(references))


def analysis_to_markdown(analysis, packet, *, model, generated_at):
    identity = packet["identity"]
    lines = [f"# HarvestGuard intervention plan: {identity['district']}", "",
             f"{identity['crop']} | Season {identity['season']} {identity['year']}",
             f"Google Gemini ({model}) | Generated {generated_at}", ""]
    for key, heading in (("meaning", "What the estimate means"), ("ranking", "Intervention priority"), ("official_briefing", "Decision brief")):
        lines.extend([f"## {heading}", "", render_text(analysis[key], packet), "", _references(analysis[key], packet), ""])
    for insight in analysis["insights"]:
        lines.extend(["## Model finding", "", render_text(insight, packet), "", _references(insight, packet), ""])
    options = {option["id"]: option for option in packet["interventions"]}
    for recommendation in analysis["recommendations"]:
        option = options[recommendation["intervention_id"]]
        lines.extend([f"## {option['title']}", "", render_text(recommendation, packet), "", _references(recommendation, packet), ""])
        if option.get("check_first"):
            lines.extend([f"Check first: {option['check_first']}", ""])
    lines.extend(["## Data and methods", ""])
    for fact in packet["facts"]:
        if fact["kind"] == "limitation":
            lines.extend([fact["text"], ""])
    lines.append("AI wording is checked against numerical facts and evidence IDs; these checks do not prove every sentence's interpretation. Recommended services are planning decisions, not measured intervention effects.")
    return "\n".join(lines) + "\n"


def _model_findings(brief, importance):
    record = brief.get("district_evidence") or {}
    findings = list(brief.get("discovered_findings", []))
    drivers = record.get("drivers", [])
    if drivers and len(findings) < 2:
        driver = drivers[0]
        direction = "higher" if driver["contribution"] > 0 else "lower"
        findings.append(("Local model comparison", driver["label"],
                         f"This is the largest sensitivity among the numeric inputs tested: the estimated loss rate is {abs(driver['contribution']):.3f} percentage points {direction} than when this input is replaced with its 2024 crop-season reference."))
    if importance and len(findings) < 2:
        labels = ", ".join(item.get("label", item["feature"]).lower() for item in importance[:3])
        findings.append(("Across the saved model", "Leading predictors", f"{labels.capitalize()} have the highest combined feature importance across the occurrence and severity models."))
    if not findings:
        findings.append(("Saved prediction", "Loss volume determines priority",
                         f"The intervention order follows estimated kg lost for {brief['crop']} in Season {brief['season']}."))
    return findings[:2]


def _finding_title(item):
    titles = {"volume_rate_comparison": "Loss intensity and volume", "district_crop_portfolio": "Across the district’s crops",
              "district_season_pattern": "The seasonal loss pattern", "loss_concentration": "Where losses concentrate", "service_timing": "When to schedule support"}
    for ref in item["evidence_ids"]:
        if ref in titles:
            return titles[ref]
        if ref.startswith("weather_"):
            return "Weather and handling conditions"
    return "Local model comparison" if item["type"] == "local_model" else "What the evidence shows"


def _render_summary(brief, packet=None, analysis=None, *, evidence_info=None, importance=None):
    quantity = brief.get("ranking_basis") == "predicted_loss_kg"
    meaning = render_text(analysis["meaning"], packet) if analysis else (
        f"Estimated loss is {brief['expected_loss_kg']:,.2f} kg, or {brief['expected_loss_pct']:.2f}% of {brief['production_kg']:,.0f} kg of covered production."
        if quantity else f"The model estimates a loss rate of {brief['expected_loss_pct']:.2f}% of production.")
    ranking = render_text(analysis["ranking"], packet) if analysis else (
        f"{brief['district']} ranks {brief['rank']} of {brief['districts_with_predictions']} by estimated kg lost."
        if quantity else f"{brief['district']} ranks {brief['rank']} of {brief['districts_with_predictions']} by predicted loss rate.")
    left, right = st.columns(2, gap="medium")
    with left:
        card("Understand", "What does it mean?", meaning, tone="green")
    with right:
        card("Compare", "Why this rank?", ranking, tone="gold")
    if brief.get("priority_districts"):
        st.markdown("### Where to intervene first")
        priority_strip(brief["priority_districts"], brief["priority_share"], selected=brief["district"])

    st.markdown("### AI insights" if analysis else "### Model findings")
    if analysis:
        st.caption("Generated automatically · Gemini · Linked to model evidence")
    findings = [("Model finding" if item["type"] in {"local_model", "global_pattern"} else "Data finding",
                 _finding_title(item), render_text(item, packet))
                for item in (analysis or {}).get("insights", []) if item["type"] not in {"limitation", "hypothesis"}][:2]
    if not findings:
        findings = _model_findings(brief, importance or [])
    columns = st.columns(min(2, len(findings)), gap="medium")
    for column, (kicker, title, text) in zip(columns, findings):
        with column:
            card(kicker, title, text)

    st.markdown("### Recommended actions")
    recommendations = {item["intervention_id"]: item for item in (analysis or {}).get("recommendations", [])}
    options = {option["id"]: option for option in brief["interventions"]}
    selected = [options[item["intervention_id"]] for item in (analysis or {}).get("recommendations", [])][:2] if recommendations else brief["interventions"][:2]
    if selected:
        columns = st.columns(min(2, len(selected)), gap="medium")
        for index, (column, option) in enumerate(zip(columns, selected), start=1):
            with column:
                wording = recommendations.get(option["id"])
                text = render_text(wording, packet) if wording else option["action"]
                source = option["source"]
                card(f"{index:02d} · {option['stage'].capitalize()}", option["title"], text,
                     evidence=option.get("rationale", option["evidence_summary"]),
                     source=("FAO · Practice evidence", source["url"]) if source.get("url") else None)
    else:
        st.write("The saved results establish the ranking; the evidence export does not select a specific service for this crop.")


def _download_explanation(brief, markdown):
    st.download_button("Download explanation", markdown.encode("utf-8"),
                       file_name=f"harvestguard_{brief['district']}_{brief['crop'].replace(' ', '_')}_{brief['season']}.md",
                       mime="text/markdown", key="download_ai")


AUTO_QUESTION = (
    "Write the intervention plan from our actual findings: explain the estimated kg and weighted loss percentage, "
    "then the district priority by kg. Give two useful insights from local numeric model comparisons, "
    "production concentration or harvest-cohort timing. Recommend the supplied actions, connecting each to its "
    "required district evidence IDs. Use direct, concise language; no generic equipment suggestions or assessment "
    "checklists. Interpret volume versus loss intensity, the district's crop portfolio and its seasonal pattern when supplied. "
    "Choose the two most decision-relevant findings and the two best-supported actions for this selection; do not repeat the headline KPIs. "
    "Explain why each selected action fits this district, using its required evidence. Technical scope and limitations are displayed separately in metadata."
)


@st.fragment(run_every=3)
def _recover_when_ready(fingerprint, retry_at, packet, api_key, model):
    cycles = st.session_state.setdefault("_ai_retry_cycles", {})
    if st.session_state.get("_ai_active") != fingerprint or cycles.get(fingerprint, 0) >= 1:
        return
    if time.time() >= retry_at and retry_explanation(packet, api_key, model=model, question=AUTO_QUESTION):
        cycles[fingerprint] = 1
        st.rerun()


@st.fragment(run_every=1)
def _watch_generation(fingerprint, future):
    if st.session_state.get("_ai_active") == fingerprint and future.done():
        st.rerun()


@st.fragment(run_every=3)
def _retry_controls(packet, saved):
    # Refresh the cooldown button even when the failure is not auto-retryable.
    # Otherwise a disabled Retry button can remain disabled until navigation.
    remaining = max(0, int(saved["retry_at"] - time.time()) + 1)
    if remaining:
        st.caption(f"Retry available in {remaining} seconds.")
    if st.button("Retry AI", key="retry_ai", disabled=remaining > 0):
        if retry_explanation(packet, _setting("GEMINI_API_KEY").strip(),
                             model=_setting("GEMINI_MODEL", DEFAULT_MODEL).strip(), question=AUTO_QUESTION):
            st.session_state.setdefault("_ai_retry_cycles", {})[st.session_state.get("_ai_active")] = 0
            st.rerun()


def _details(brief, packet, saved, evidence_info=None):
    with st.expander("AI explanation & evidence"):
        if saved and "analysis" in saved:
            st.caption(f"Google Gemini ? {saved['model']} ? Generated {saved['generated_at']}" + (" ? Saved explanation" if saved.get("cached") else ""))
        elif saved and "error" in saved:
            st.write(saved["error"])
            if saved.get("validation_issue"):
                st.caption("Evidence check: " + saved["validation_issue"])
            st.caption(f"Generation status: {saved['code']}" + (f" (HTTP {saved['http_status']})" if saved.get("http_status") else "") + ". Showing verified model findings.")
            _retry_controls(packet, saved)
        else:
            st.caption("Showing verified model findings. AI wording appears when a checked explanation is available.")
        used = {ref for section in ((saved or {}).get("analysis", {}).get("insights", []) + (saved or {}).get("analysis", {}).get("recommendations", [])) for ref in section["evidence_ids"]}
        sources = {fact["source_url"]: fact.get("source_title", "Source") for fact in (packet or {}).get("facts", [])
                   if fact.get("source_url") and (fact["id"] in used or fact["kind"] == "intervention_evidence")}
        for url, title in sources.items():
            st.markdown(f"[{title}]({url})")
        if packet:
            st.download_button("Download district evidence", json.dumps(packet, indent=2, ensure_ascii=False).encode("utf-8"),
                               file_name=f"harvestguard_{brief['district']}_evidence.json", mime="application/json", key="download_ai_evidence")


def render_ai_advisor(brief, view, app_dir, *, model_dir=None):
    if brief is None:
        return
    model_dir = model_dir or app_dir / MODEL_FOLDER
    evidence_info, importance, packet = None, [], None
    try:
        record, evidence_info, importance = load_insight_evidence(app_dir, model_dir, brief)
        brief = plan_interventions(brief, view.to_dict("records"), record,
                                  comparative_records=evidence_info.get("comparative_records") if evidence_info else None)
        local = ({**record, "expected_loss_pct": brief["expected_loss_pct"], "method": evidence_info["method"], "aggregation": evidence_info["aggregation"]}
                 if record and record["drivers"] else load_local_explanation(model_dir / "local_explanations.json", brief))
        packet = build_evidence_packet(brief, view.to_dict("records"), global_importance=importance,
                                       context=load_context(app_dir / "knowledge_base" / "district_context.json"), local_explanation=local)
    except (OSError, ValueError, KeyError, TypeError):
        brief = plan_interventions(brief, view.to_dict("records"))
        _render_summary(brief)
        _details(brief, None, None)
        _download_explanation(brief, brief_to_markdown(brief))
        return
    api_key = _setting("GEMINI_API_KEY").strip() if _setting("HARVESTGUARD_AI_ENABLED", "1").strip() != "0" else ""
    model = _setting("GEMINI_MODEL", DEFAULT_MODEL).strip()
    _sync_configured_key(api_key)
    saved = None
    if api_key:
        fingerprint = request_fingerprint(packet, model, AUTO_QUESTION)
        st.session_state["_ai_active"] = fingerprint
        cache = st.session_state.setdefault("_ai_cache", {})
        saved = cache.get(fingerprint) or lookup_explanation(packet, api_key, model=model, question=AUTO_QUESTION)
        if saved is None:
            future = request_explanation(packet, api_key, model=model, question=AUTO_QUESTION, generator=generate_analysis)
            if future.done():
                saved = future.result()
            else:
                _watch_generation(fingerprint, future)
        if saved and "analysis" in saved:
            cache[fingerprint] = saved
            while len(cache) > 64:
                del cache[next(iter(cache))]
        elif saved and saved["retryable"] and saved["retry_at"] <= time.time() + 120:
            _recover_when_ready(fingerprint, saved["retry_at"], packet, api_key, model)
    analysis = (saved or {}).get("analysis")
    _render_summary(brief, packet, analysis, evidence_info=evidence_info, importance=importance)
    _details(brief, packet, saved, evidence_info)
    markdown = analysis_to_markdown(analysis, packet, model=saved["model"], generated_at=saved["generated_at"]) if analysis else brief_to_markdown({**brief, "interventions": brief["interventions"][:2]})
    _download_explanation(brief, markdown)

