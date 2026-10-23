import json
import io
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from copy import deepcopy
from pathlib import Path
from urllib.error import HTTPError, URLError
import ssl

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import harvestguard_ai
import harvestguard_ai_view
import harvestguard_ai_cache
from harvestguard_ai import (
    AIError, build_evidence_packet, check_connection, generate_analysis, load_context,
    load_local_explanation, render_text, request_fingerprint, validate_analysis, ground_numeric_references,
    normalize_response_shape, AIResponseError,
)
from harvestguard_ai_view import analysis_to_markdown
from harvestguard_decisions import build_brief, load_knowledge_base


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def inputs():
    data = pd.read_csv(ROOT / "model_v2" / "harvestguard_2025_v2_vulnerability.csv")
    view = data.loc[(data.season == "A") & (data.crop_name == "Maize")].copy()
    view["display_rank"] = view.predicted_loss_pct.rank(method="first", ascending=False).astype(int)
    row = view.loc[view.district == "Nyagatare"].iloc[0].to_dict()
    kb = load_knowledge_base(ROOT / "knowledge_base" / "interventions.json")
    metadata = json.loads((ROOT / "model_v2" / "model_v2_metadata.json").read_text(encoding="utf-8"))
    brief = build_brief(row, metadata, kb, district_count=len(view))
    return brief, view.to_dict("records")


@pytest.fixture
def packet(inputs):
    brief, peers = inputs
    return build_evidence_packet(
        brief, peers,
        context=load_context(ROOT / "knowledge_base" / "district_context.json"),
        global_importance=[{"feature": "cat__district_Nyagatare", "combined_importance": 0.112632}],
    )


def valid_response(packet):
    """Transport fixture, never used as a simulated live response in the app."""
    def statement(text, refs):
        return {"text": text, "evidence_ids": refs}
    recommendations = []
    if packet["interventions"]:
        option = packet["interventions"][0]
        recommendations.append({
            **statement("Prioritize the supplied service using the matching district findings.", ["intervention_" + option["id"], *option.get("required_evidence_ids", [])]),
            "intervention_id": option["id"],
        })
    return {
        "identity": deepcopy(packet["identity"]),
        "meaning": statement("The estimated loss is {{loss_kg}}, corresponding to {{loss_pct}} of the covered production.", ["prediction", "quantity"]) if "loss_kg" in packet["display_values"] else statement("The expected rate is {{loss_pct}}, illustrated as {{kg_per_tonne}}. This is not a district loss total.", ["prediction", "scale"]),
        "ranking": statement("The district is ranked {{rank}} among {{district_count}} districts by estimated kg lost; the comparison median is {{median_loss_kg}}.", ["ranking"]) if "loss_kg" in packet["display_values"] else statement("The district is ranked {{rank}} among {{district_count}} districts; the comparison median is {{median_loss_pct}}.", ["ranking"]),
        "insights": [], "recommendations": recommendations,
        "next_steps": statement("Ask an agronomist to verify local conditions and available evidence.", ["limitations"]),
        "official_briefing": statement("Use the expected rate to guide a field assessment, then select a suitable pilot from the evidence.", ["prediction", "limitations"]),
    }


def test_packet_joins_only_matching_context_and_converts_percent_correctly(inputs, packet):
    brief, peers = inputs
    assert packet["display_values"]["kg_per_tonne"] == "5.79 kg per tonne"
    assert packet["display_values"]["loss_pct"] == "0.58%"
    assert any(fact["kind"] == "published_context" for fact in packet["facts"])
    other = {**brief, "season": "B"}
    other_peers = [{**row, "season": "B"} for row in peers]
    other_packet = build_evidence_packet(other, other_peers, context=load_context(ROOT / "knowledge_base" / "district_context.json"))
    assert not any(fact["kind"] == "published_context" for fact in other_packet["facts"])
    specific = {**brief, "expected_loss_pct": 0.4}
    matching = [{**row, "predicted_loss_pct": 0.4} if row["district"] == brief["district"] else row for row in peers]
    assert build_evidence_packet(specific, matching)["display_values"]["kg_per_tonne"] == "4.00 kg per tonne"


def test_validated_response_renders_trusted_numbers_and_exports_sources(packet):
    response = validate_analysis(valid_response(packet), packet)
    assert "5.79 kg per tonne" in render_text(response["meaning"], packet)
    markdown = analysis_to_markdown(response, packet, model="test-model", generated_at="test-time")
    assert "https://www.fao.org/" in markdown
    assert "Check first:" in markdown
    assert "{{" not in markdown
    assert "do not prove every sentence" in markdown


def test_exact_known_numbers_from_live_style_prose_are_grounded(packet):
    response = valid_response(packet)
    response["meaning"]["text"] = "The 2025 expected rate is 0.58%, illustrated as 5.79 kg per tonne."
    response["insights"] = [{
        "type": "context", "text": "NISR reports 85,407 metric tonnes as production context.",
        "evidence_ids": ["context_nisr_2025_a_nyagatare_maize"],
    }]
    grounded = ground_numeric_references(response, packet)
    validate_analysis(grounded, packet)
    assert render_text(grounded["meaning"], packet) == response["meaning"]["text"]
    assert render_text(grounded["insights"][0], packet) == response["insights"][0]["text"]
    assert "{{" in grounded["meaning"]["text"]


def test_numerical_grounding_does_not_accept_unknown_values_or_local_causes(packet):
    response = valid_response(packet)
    response["meaning"]["text"] = "Expected losses reach 40%."
    with pytest.raises(AIError):
        validate_analysis(ground_numeric_references(response, packet), packet)
    response = valid_response(packet)
    response["insights"] = [{"type": "local_model", "text": "The local model caused the ranking.", "evidence_ids": ["explanation_status"]}]
    with pytest.raises(AIError):
        validate_analysis(ground_numeric_references(response, packet), packet)


def test_harmless_response_variations_keep_the_same_evidence(packet):
    response = valid_response(packet)
    response.pop("identity")
    response["meaning"]["title"] = "Extra display title"
    response["meaning"]["evidence_ids"] = " PREDICTION "
    normalized = normalize_response_shape(response, packet)
    validate_analysis(normalized, packet)
    assert normalized["identity"] == packet["identity"]
    assert normalized["meaning"]["evidence_ids"] == ["prediction"]
    assert "title" not in normalized["meaning"]
    response["meaning"]["evidence_ids"] = "invented source"
    with pytest.raises(AIError):
        validate_analysis(normalize_response_shape(response, packet), packet)


def test_assessment_order_follows_evidence_prerequisites(packet):
    response = valid_response(packet)
    response["recommendations"] = [
        {"text": "Assess this option after checking its conditions.",
         "evidence_ids": ["intervention_" + option["id"]], "intervention_id": option["id"]}
        for option in reversed(packet["interventions"])
    ]
    result = validate_analysis(response, packet)
    assert [item["intervention_id"] for item in result["recommendations"]] == [option["id"] for option in packet["interventions"]]


def test_comparison_does_not_invent_significance_or_add_harvesting_to_target(packet):
    response = valid_response(packet)
    response["ranking"]["text"] = "The expected loss is significantly higher than the comparison median."
    with pytest.raises(AIResponseError, match="overstated"):
        validate_analysis(response, packet)
    response = valid_response(packet)
    response["next_steps"]["text"] = "Inspect losses at the harvest, processing, or storage stages."
    with pytest.raises(AIResponseError, match="outside"):
        validate_analysis(response, packet)


def test_invalid_citation_is_corrected_once_using_original_evidence(packet, monkeypatch):
    calls = []
    def reply(api_key, model, body):
        request = json.loads(body["contents"][0]["parts"][0]["text"])
        calls.append(request)
        draft = valid_response(packet)
        if len(calls) == 1:
            draft["meaning"]["evidence_ids"] = ["invented_source"]
        return {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": json.dumps(draft)}]}}]}
    monkeypatch.setattr(harvestguard_ai, "_call_gemini", reply)
    result = generate_analysis(packet, "test-key-only")
    assert len(calls) == 2
    assert calls[0]["repair"] is None
    assert "evidence_ids" in calls[1]["repair"]["validation_issue"]
    assert calls[1]["evidence_packet"] == packet
    assert "test-key-only" not in json.dumps(calls)
    assert result["meaning"]["evidence_ids"] == ["prediction", "scale"]


def test_invalid_repair_stops_after_two_requests(packet, monkeypatch):
    calls = []
    def reply(api_key, model, body):
        calls.append(body)
        draft = valid_response(packet)
        draft["recommendations"][0]["intervention_id"] = "unsupported_action"
        return {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": json.dumps(draft)}]}}]}
    monkeypatch.setattr(harvestguard_ai, "_call_gemini", reply)
    with pytest.raises(AIResponseError, match="after an automatic correction"):
        generate_analysis(packet, "test-key-only")
    assert len(calls) == 2


@pytest.mark.parametrize("corruption", ["prediction", "citation", "raw_number", "placeholder", "local_driver", "intervention"])
def test_unsupported_ai_claims_are_rejected(packet, corruption):
    response = valid_response(packet)
    if corruption == "prediction":
        response["identity"]["expected_loss_pct"] = 99
    elif corruption == "citation":
        response["meaning"]["evidence_ids"] = ["invented_source"]
    elif corruption == "raw_number":
        response["meaning"]["text"] = "The loss is 40%."
    elif corruption == "placeholder":
        response["meaning"]["text"] = "The loss is {{invented_loss}}."
    elif corruption == "local_driver":
        response["insights"] = [{"type": "local_model", "text": "A district feature explains this prediction.", "evidence_ids": ["global_0"]}]
    elif corruption == "intervention":
        response["recommendations"][0]["intervention_id"] = "invented_intervention"
    with pytest.raises(AIError):
        validate_analysis(response, packet)


def test_user_context_needs_source_and_date_and_is_fingerprinted(inputs, packet):
    brief, peers = inputs
    with pytest.raises(ValueError, match="source, date"):
        build_evidence_packet(brief, peers, field_note={"text": "Wet grain was observed."})
    with pytest.raises(ValueError, match="selected year"):
        build_evidence_packet(brief, peers, field_note={"text": "Wet grain was observed.", "source": "Agronomist visit", "date": "2026-10-21", "period": "historical"})
    follow_up = build_evidence_packet(brief, peers, field_note={"text": "Wet grain was observed.", "source": "Agronomist visit", "date": "2026-10-21", "period": "follow_up"})
    assert next(f for f in follow_up["facts"] if f["id"] == "field_note")["period"] == "follow_up"
    assert request_fingerprint(packet, "model", "") != request_fingerprint(follow_up, "model", "")


def test_no_crop_evidence_means_no_generated_intervention(inputs):
    brief, peers = inputs
    brief = {**brief, "crop": "Cassava", "interventions": []}
    peers = [{**row, "crop_name": "Cassava"} for row in peers]
    packet = build_evidence_packet(brief, peers)
    response = valid_response(packet)
    assert validate_analysis(response, packet)["recommendations"] == []
    response["recommendations"] = [{"text": "Try a new intervention.", "evidence_ids": ["prediction"], "intervention_id": "invented"}]
    with pytest.raises(AIError):
        validate_analysis(response, packet)


def test_no_key_does_not_send_any_network_request(packet, monkeypatch):
    def unexpected_request(*_, **__):
        pytest.fail("No request should occur without a key.")
    monkeypatch.setattr(harvestguard_ai, "urlopen", unexpected_request)
    with pytest.raises(AIError, match="API key"):
        generate_analysis(packet, "")


def test_local_model_export_must_match_saved_prediction(inputs, tmp_path):
    brief, peers = inputs
    record = {key: brief[key] for key in ("district", "crop", "season", "year", "expected_loss_pct")}
    record.update({"method": "Test attribution", "aggregation": "Documented test aggregation", "drivers": [{"feature": "test_feature", "component": "occurrence", "unit": "log-odds", "contribution": 0.1}]})
    path = tmp_path / "explanation.json"
    path.write_text(json.dumps({"schema_version": 1, "records": [record]}), encoding="utf-8")
    explanation = load_local_explanation(path, brief)
    packet = build_evidence_packet(brief, peers, local_explanation=explanation)
    assert any(fact["kind"] == "local_model" for fact in packet["facts"])
    record["expected_loss_pct"] = 40
    path.write_text(json.dumps({"schema_version": 1, "records": [record]}), encoding="utf-8")
    with pytest.raises(ValueError, match="different saved prediction"):
        load_local_explanation(path, brief)


def test_gemini_transport_sends_structured_packet_without_key_in_body(packet, monkeypatch):
    expected = valid_response(packet)
    def fake_urlopen(request, timeout):
        assert timeout == 30
        assert "test-secret" not in request.full_url
        body = json.loads(request.data)
        assert "test-secret" not in json.dumps(body)
        assert body["generationConfig"]["responseMimeType"] == "application/json"
        assert body["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "MINIMAL"
        assert body["generationConfig"]["responseSchema"]["type"] == "OBJECT"
        assert "additionalProperties" not in json.dumps(body["generationConfig"]["responseSchema"])
        assert "responseFormat" not in body["generationConfig"]
        assert json.loads(body["contents"][0]["parts"][0]["text"])["evidence_packet"]["identity"] == packet["identity"]
        class Response:
            def __enter__(self):
                return self
            def __exit__(self, *_):
                pass
            def read(self, _):
                return json.dumps({"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": json.dumps(expected)}]}}]}).encode()
        return Response()
    monkeypatch.setattr(harvestguard_ai, "urlopen", fake_urlopen)
    assert generate_analysis(packet, "test-secret") == expected


@pytest.mark.parametrize("code", [400, 403, 404, 429, 503])
def test_provider_errors_do_not_expose_keys(packet, monkeypatch, code):
    def fail(*_, **__):
        raise HTTPError("https://example.com/test-secret", code, "test-secret", None, None)
    monkeypatch.setattr(harvestguard_ai, "urlopen", fail)
    with pytest.raises(AIError) as error:
        generate_analysis(packet, "test-secret")
    assert "test-secret" not in str(error.value)


def test_project_access_denial_is_distinguished_without_echoing_provider_body(packet, monkeypatch):
    def fail(*_, **__):
        body = {"error": {"message": "Your project has been denied access. Please contact support.", "debug_key": "test-secret"}}
        raise HTTPError("https://example.com", 403, "Forbidden", None, io.BytesIO(json.dumps(body).encode()))
    monkeypatch.setattr(harvestguard_ai, "urlopen", fail)
    with pytest.raises(AIError, match="Google has blocked the project") as error:
        generate_analysis(packet, "test-secret")
    assert "test-secret" not in str(error.value)
    assert "contact Google support" in str(error.value)


def test_connection_check_sends_only_a_tiny_non_agricultural_prompt(monkeypatch):
    def fake_urlopen(request, timeout):
        body = json.loads(request.data)
        assert timeout == 30
        assert "evidence_packet" not in json.dumps(body)
        assert body["contents"][0]["parts"][0]["text"] == "Reply only OK."
        class Response:
            def __enter__(self):
                return self
            def __exit__(self, *_):
                pass
            def read(self, _):
                return b'{"candidates":[{"finishReason":"STOP","content":{"parts":[{"text":"OK"}]}}]}'
        return Response()
    monkeypatch.setattr(harvestguard_ai, "urlopen", fake_urlopen)
    assert "accepted a live request" in check_connection("test-secret")


def test_private_settings_are_found_independently_of_shell_directory(tmp_path, monkeypatch):
    settings = tmp_path / "secrets.toml"
    settings.write_text('GEMINI_API_KEY = "private-test-key"\n', encoding="utf-8")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(harvestguard_ai_view, "PRIVATE_SETTINGS", settings)
    assert harvestguard_ai_view._setting("GEMINI_API_KEY") == "private-test-key"






def test_malformed_provider_content_fails_with_readable_error(packet, monkeypatch):
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
        def read(self, _):
            return b'{"candidates":[{"finishReason":"STOP","content":null}]}'
    monkeypatch.setattr(harvestguard_ai, "urlopen", lambda *args, **kwargs: Response())
    with pytest.raises(AIError, match="readable explanation"):
        generate_analysis(packet, "test-secret")






def test_insights_generate_automatically_and_reuse_each_selection(monkeypatch):
    calls = []
    monkeypatch.setattr(harvestguard_ai_view, "_setting", lambda name, default="": "private-test-key" if name == "GEMINI_API_KEY" else default)
    def generate(packet, api_key, **kwargs):
        assert api_key == "private-test-key"
        calls.append(packet["identity"])
        return validate_analysis(valid_response(packet), packet)
    monkeypatch.setattr(harvestguard_ai_view, "generate_analysis", generate)
    app = AppTest.from_file(str(ROOT / "harvestguard_app.py")).run(timeout=40)
    assert not calls
    app.selectbox(key="brief_district").select("Nyagatare").run(timeout=40)
    assert not app.text_input
    assert not any("generate" in button.label.lower() for button in app.button)
    app.radio(key="page").set_value("Insights").run(timeout=40)
    assert not app.exception, [item.message for item in app.exception]
    assert len(calls) == 1
    assert calls[0]["district"] == "Nyagatare"
    assert "64,545.97 kg" in "\n".join(item.value for item in app.markdown)
    app.run(timeout=40)
    assert len(calls) == 1
    app.selectbox(key="brief_district").select("Kamonyi").run(timeout=40)
    assert not app.exception
    assert len(calls) == 2
    assert calls[-1]["district"] == "Kamonyi"
    app.selectbox(key="brief_district").select("Nyagatare").run(timeout=40)
    app.radio(key="page").set_value("Map").run(timeout=40)
    app.radio(key="page").set_value("Insights").run(timeout=40)
    assert len(calls) == 2
    app.selectbox(key="crop").select("Tomato").run(timeout=40)
    assert not app.exception
    assert calls[-1]["crop"] == "Tomato"
    assert "Recommended actions" in "\n".join(item.value for item in app.markdown)
    assert "protective, ventilated packaging" not in "\n".join(item.value for item in app.markdown)
    app.selectbox(key="crop").select("Cassava").run(timeout=40)
    assert not app.exception
    assert "Recommended actions" in "\n".join(item.value for item in app.markdown)
    app.selectbox(key="season").select("B").run(timeout=40)
    assert not app.exception
    assert calls[-1]["season"] == "B"
    assert "private-test-key" not in "\n".join(item.value for item in app.markdown)


def test_replacing_private_key_regenerates_with_no_old_session_override(monkeypatch):
    config = {"key": "original-test-key"}
    calls = []
    monkeypatch.setattr(harvestguard_ai_view, "_setting", lambda name, default="": config["key"] if name == "GEMINI_API_KEY" else default)
    def generate(packet, api_key, **kwargs):
        calls.append(api_key)
        return valid_response(packet)
    monkeypatch.setattr(harvestguard_ai_view, "generate_analysis", generate)
    app = AppTest.from_file(str(ROOT / "harvestguard_app.py")).run(timeout=40)
    app.session_state["_gemini_key"] = "old-manual-override"
    app.radio(key="page").set_value("Insights").run(timeout=40)
    assert calls == ["original-test-key"]
    config["key"] = "replacement-test-key"
    app.run(timeout=40)
    assert not app.exception
    assert calls == ["original-test-key", "replacement-test-key"]
    assert "_gemini_key" not in app.session_state


def test_unverified_response_shows_evidence_and_does_not_retry_on_rerun(monkeypatch):
    calls = []
    clock = {"now": 1000}
    monkeypatch.setattr(harvestguard_ai_cache, "time", SimpleNamespace(time=lambda: clock["now"]))
    monkeypatch.setattr(harvestguard_ai_view, "time", SimpleNamespace(time=lambda: clock["now"]))
    monkeypatch.setattr(harvestguard_ai_view, "_setting", lambda name, default="": "private-test-key" if name == "GEMINI_API_KEY" else default)
    def generate(*args, **kwargs):
        calls.append(True)
        raise AIResponseError("The generated explanation could not be verified after an automatic correction.", "Test issue")
    monkeypatch.setattr(harvestguard_ai_view, "generate_analysis", generate)
    app = AppTest.from_file(str(ROOT / "harvestguard_app.py")).run(timeout=40)
    app.radio(key="page").set_value("Insights").run(timeout=40)
    assert not app.exception
    assert "### Model findings" in "\n".join(item.value for item in app.markdown)
    assert not any("Generated automatically" in item.value for item in app.caption)
    app.run(timeout=40)
    assert len(calls) == 1
    assert app.button(key="retry_ai").disabled
    clock["now"] += 61
    app.run(timeout=40)
    assert not app.exception
    assert len(calls) == 1
    assert not app.button(key="retry_ai").disabled
    app.button(key="retry_ai").click().run(timeout=40)
    assert not app.exception
    assert len(calls) == 2
    app.selectbox(key="brief_district").select("Kamonyi").run(timeout=40)
    assert len(calls) == 3
    assert "Kamonyi: from insight to action" in "\n".join(item.value for item in app.markdown)


def test_transient_transport_error_retries_once_and_recovers(packet, monkeypatch):
    calls, waits = [], []
    monkeypatch.setattr(harvestguard_ai, "time", SimpleNamespace(monotonic=lambda: 100, sleep=waits.append))
    def request(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise HTTPError("https://example.com", 503, "secret", {}, None)
        class Response:
            def __enter__(self):
                return self
            def __exit__(self, *_):
                pass
            def read(self, _):
                return json.dumps({"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": json.dumps(valid_response(packet))}]}}]}).encode()
        return Response()
    monkeypatch.setattr(harvestguard_ai, "urlopen", request)
    assert generate_analysis(packet, "secret")["identity"] == packet["identity"]
    assert len(calls) == 2
    assert len(waits) == 1 and waits[0] >= 1


def test_provider_retry_after_is_respected_without_blocking_ui(packet, monkeypatch):
    calls, waits = [], []
    monkeypatch.setattr(harvestguard_ai, "time", SimpleNamespace(monotonic=lambda: 100, sleep=waits.append))
    def request(*args, **kwargs):
        calls.append(True)
        detail = {"error": {"details": [{"retryDelay": "45s"}]}}
        raise HTTPError("https://example.com", 429, "secret", {"Retry-After": "30"}, io.BytesIO(json.dumps(detail).encode()))
    monkeypatch.setattr(harvestguard_ai, "urlopen", request)
    with pytest.raises(AIError) as caught:
        generate_analysis(packet, "secret")
    assert caught.value.retry_after == 45
    assert caught.value.retryable
    assert caught.value.code == "rate_limit"
    assert len(calls) == 1 and not waits


def test_daily_quota_and_project_denial_are_not_retried(packet, monkeypatch):
    calls = []
    def request(*args, **kwargs):
        calls.append(True)
        body = {"error": {"details": [{"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}}
        raise HTTPError("https://example.com", 429, "secret", {}, io.BytesIO(json.dumps(body).encode()))
    monkeypatch.setattr(harvestguard_ai, "urlopen", request)
    with pytest.raises(AIError) as caught:
        generate_analysis(packet, "secret")
    assert caught.value.code == "daily_quota"
    assert not caught.value.retryable and len(calls) == 1


def test_validated_disk_cache_survives_session_and_provider_failure(packet):
    calls = []
    def generate(*args, **kwargs):
        calls.append(True)
        return valid_response(packet)
    first = harvestguard_ai_cache.get_explanation(packet, "private-test-key", model="test-model", question="short", generator=generate)
    def offline(*args, **kwargs):
        pytest.fail("An exact saved explanation should not require Google.")
    second = harvestguard_ai_cache.get_explanation(packet, "private-test-key", model="test-model", question="short", generator=offline)
    assert first["analysis"] == second["analysis"]
    assert second["cached"] and len(calls) == 1
    saved = list(harvestguard_ai_cache.CACHE_DIR.rglob("*.json"))
    assert len(saved) == 1
    assert "private-test-key" not in saved[0].read_text(encoding="utf-8")


def test_changed_evidence_and_key_cannot_reuse_other_answers(packet):
    calls = []
    def generate(evidence, *args, **kwargs):
        calls.append(True)
        return valid_response(evidence)
    harvestguard_ai_cache.get_explanation(packet, "first-key", model="test-model", question="short", generator=generate)
    changed = deepcopy(packet)
    changed["prompt_version"] += "-updated"
    harvestguard_ai_cache.get_explanation(changed, "first-key", model="test-model", question="short", generator=generate)
    harvestguard_ai_cache.get_explanation(packet, "replacement-key", model="test-model", question="short", generator=generate)
    assert len(calls) == 3


@pytest.mark.parametrize("corruption", ["invalid_json", "invalid_answer", "list"])
def test_bad_disk_cache_is_replaced_with_verified_answer(packet, corruption):
    calls = []
    def generate(*args, **kwargs):
        calls.append(True)
        return valid_response(packet)
    harvestguard_ai_cache.get_explanation(packet, "test-key", model="test-model", question="short", generator=generate)
    path = next(harvestguard_ai_cache.CACHE_DIR.rglob("*.json"))
    if corruption == "invalid_answer":
        data = json.loads(path.read_text(encoding="utf-8"))
        data["analysis"]["meaning"]["text"] = "Losses are 99%."
        path.write_text(json.dumps(data), encoding="utf-8")
    else:
        path.write_text("invalid" if corruption == "invalid_json" else "[]", encoding="utf-8")
    result = harvestguard_ai_cache.get_explanation(packet, "test-key", model="test-model", question="short", generator=generate)
    assert "analysis" in result and len(calls) == 2


def test_cooldown_expires_and_covers_other_selections(packet, monkeypatch):
    clock, calls = {"now": 1000}, []
    monkeypatch.setattr(harvestguard_ai_cache, "time", SimpleNamespace(time=lambda: clock["now"]))
    def generate(evidence, *args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise AIError("Rate limit reached.", code="rate_limit", retryable=True, retry_after=30, http_status=429)
        return valid_response(evidence)
    first = harvestguard_ai_cache.get_explanation(packet, "test-key", model="test-model", question="one", generator=generate)
    other = harvestguard_ai_cache.get_explanation(packet, "test-key", model="test-model", question="two", generator=generate)
    assert first["retry_at"] == other["retry_at"] == 1030
    assert len(calls) == 1
    clock["now"] = 1031
    recovered = harvestguard_ai_cache.get_explanation(packet, "test-key", model="test-model", question="two", generator=generate)
    assert "analysis" in recovered and len(calls) == 2


def test_two_sessions_share_one_inflight_request(packet):
    entered, release = Event(), Event()
    calls = []
    def generate(*args, **kwargs):
        calls.append(True)
        entered.set()
        assert release.wait(timeout=5)
        return valid_response(packet)
    def request():
        return harvestguard_ai_cache.get_explanation(packet, "test-key", model="test-model", question="short", generator=generate)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(request)
        assert entered.wait(timeout=5)
        second = executor.submit(request)
        release.set()
        a, b = first.result(timeout=5), second.result(timeout=5)
    assert a["analysis"] == b["analysis"] and len(calls) == 1


def test_background_request_keeps_ui_responsive_and_shares_work(packet, monkeypatch):
    entered, release = Event(), Event()
    def generate(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=5)
        return valid_response(packet)
    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(harvestguard_ai_cache, "_executor", executor)
        try:
            future = harvestguard_ai_cache.request_explanation(packet, "test-key", model="test-model", question="short", generator=generate)
            assert entered.wait(timeout=5)
            assert not future.done()
            repeated = harvestguard_ai_cache.request_explanation(packet, "test-key", model="test-model", question="short", generator=generate)
            assert repeated is future
        finally:
            release.set()
        assert "analysis" in future.result(timeout=5)


def test_ui_recovers_after_rate_limit_and_summary_tracks_new_selection(monkeypatch):
    clock, calls = {"now": 1000}, []
    monkeypatch.setattr(harvestguard_ai_cache, "time", SimpleNamespace(time=lambda: clock["now"]))
    monkeypatch.setattr(harvestguard_ai_view, "time", SimpleNamespace(time=lambda: clock["now"]))
    monkeypatch.setattr(harvestguard_ai_view, "_setting", lambda name, default="": "test-key" if name == "GEMINI_API_KEY" else default)
    def generate(packet, *args, **kwargs):
        calls.append(packet["identity"]["district"])
        if len(calls) == 1:
            raise AIError("Google request limit reached.", code="rate_limit", retryable=True, retry_after=30, http_status=429)
        return valid_response(packet)
    monkeypatch.setattr(harvestguard_ai_view, "generate_analysis", generate)
    app = AppTest.from_file(str(ROOT / "harvestguard_app.py")).run(timeout=40)
    app.radio(key="page").set_value("Insights").run(timeout=40)
    assert not app.exception
    assert len(calls) == 1
    app.selectbox(key="brief_district").select("Kamonyi").run(timeout=40)
    assert not app.exception
    assert len(calls) == 1
    assert "Kamonyi ranks" in "\n".join(item.value for item in app.markdown)
    clock["now"] += 31
    app.run(timeout=40)
    assert not app.exception
    assert calls == ["Nyagatare", "Kamonyi"]
    assert any("Generated automatically" in item.value for item in app.caption)


def test_provider_schema_stays_compact_with_full_local_evidence_validation(packet):
    schema = harvestguard_ai.provider_response_schema(packet)
    assert "identity" not in schema["properties"]
    assert "identity" not in schema["required"]
    assert "enum" not in schema["properties"]["meaning"]["properties"]["evidence_ids"]["items"]
    assert harvestguard_ai.response_contract(packet)["allowed_evidence_ids"] == [fact["id"] for fact in packet["facts"]]
    assert harvestguard_ai.response_schema(packet)["properties"]["meaning"]["properties"]["evidence_ids"]["items"]["enum"] == [fact["id"] for fact in packet["facts"]]
    assert schema["properties"]["recommendations"]["items"]["properties"]["intervention_id"]["enum"] == [option["id"] for option in packet["interventions"]]


def test_more_evidence_does_not_expand_the_provider_grammar(packet):
    expanded = deepcopy(packet)
    expanded["facts"].extend({"id": f"additional_fact_{i}", "kind": "limitation", "text": "Synthetic schema stress-test fact."} for i in range(100))
    assert harvestguard_ai.provider_response_schema(expanded) == harvestguard_ai.provider_response_schema(packet)
    assert len(harvestguard_ai.response_contract(expanded)["allowed_evidence_ids"]) == len(packet["facts"]) + 100


def test_interface_displays_the_actions_selected_by_ai(monkeypatch):
    chosen, unused = [], []
    monkeypatch.setattr(harvestguard_ai_view, "_setting", lambda name, default="": "test-key" if name == "GEMINI_API_KEY" else default)
    def generate(packet, *args, **kwargs):
        assert len(packet["interventions"]) >= 3
        option = packet["interventions"][-1]
        chosen.append(option["title"])
        unused.append(packet["interventions"][0]["title"])
        response = valid_response(packet)
        response["recommendations"] = [{"intervention_id": option["id"], "text": "Schedule the supplied service using the matching district findings.",
                                        "evidence_ids": ["intervention_" + option["id"], *option["required_evidence_ids"]]}]
        return validate_analysis(response, packet)
    monkeypatch.setattr(harvestguard_ai_view, "generate_analysis", generate)
    app = AppTest.from_file(str(ROOT / "harvestguard_app.py")).run(timeout=40)
    app.radio(key="page").set_value("Insights").run(timeout=40)
    assert not app.exception
    text = "\n".join(item.value for item in app.markdown)
    assert chosen[0] in text
    assert unused[0] not in text


def test_failed_selection_requires_explicit_retry_after_cooldown(packet, monkeypatch):
    clock, calls = {"now": 1000}, []
    monkeypatch.setattr(harvestguard_ai_cache, "time", SimpleNamespace(time=lambda: clock["now"]))
    def generate(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise AIResponseError("The explanation failed an evidence check.", "Use the supplied evidence IDs.")
        return valid_response(packet)
    def request():
        return harvestguard_ai_cache.request_explanation(packet, "test-key", model="test-model", question="short", generator=generate)
    first = request()
    assert not harvestguard_ai_cache.retry_explanation(packet, "test-key", model="test-model", question="short")
    clock["now"] += 3600
    assert request() is first
    assert len(calls) == 1
    assert first.result()["validation_issue"] == "Use the supplied evidence IDs."
    assert harvestguard_ai_cache.retry_explanation(packet, "test-key", model="test-model", question="short")
    assert "analysis" in request().result()
    assert len(calls) == 2


def test_successful_answer_is_reused_when_disk_is_unwritable(packet, monkeypatch):
    calls = []
    monkeypatch.setattr(harvestguard_ai_cache, "_write", lambda *args: False)
    def generate(*args, **kwargs):
        calls.append(True)
        return valid_response(packet)
    first = harvestguard_ai_cache.get_explanation(packet, "test-key", model="test-model", question="short", generator=generate)
    second = harvestguard_ai_cache.get_explanation(packet, "test-key", model="test-model", question="short", generator=generate)
    assert first["analysis"] == second["analysis"]
    assert second["cached"] and len(calls) == 1


def test_unexpected_background_error_preserves_dashboard_and_hides_secrets(packet, caplog):
    def generate(*args, **kwargs):
        raise RuntimeError("private-test-secret")
    future = harvestguard_ai_cache.request_explanation(packet, "test-key", model="test-model", question="short", generator=generate)
    result = future.result()
    assert result["code"] == "internal"
    assert "model results remain available" in result["error"]
    assert "private-test-secret" not in json.dumps(result) + caplog.text


@pytest.mark.parametrize("reason,code", [
    (PermissionError("test-secret"), "network_blocked"),
    (ssl.SSLCertVerificationError("test-secret"), "certificate"),
])
def test_persistent_network_configuration_failures_are_not_retried(monkeypatch, reason, code):
    calls = []
    def fail(*args, **kwargs):
        calls.append(True)
        raise URLError(reason)
    monkeypatch.setattr(harvestguard_ai, "urlopen", fail)
    with pytest.raises(AIError) as caught:
        check_connection("test-secret")
    assert caught.value.code == code
    assert not caught.value.retryable and len(calls) == 1
    assert "test-secret" not in str(caught.value)


def test_persistent_service_outage_has_bounded_exponential_retries(monkeypatch):
    calls, waits = [], []
    monkeypatch.setattr(harvestguard_ai, "time", SimpleNamespace(monotonic=lambda: 100, sleep=waits.append))
    def fail(*args, **kwargs):
        calls.append(True)
        raise HTTPError("https://example.com", 503, "test-secret", {}, None)
    monkeypatch.setattr(harvestguard_ai, "urlopen", fail)
    with pytest.raises(AIError) as caught:
        check_connection("test-secret")
    assert caught.value.code == "provider_busy" and caught.value.http_status == 503
    assert len(calls) == 3
    assert len(waits) == 2 and 1 <= waits[0] < waits[1]


def test_connection_check_does_not_treat_empty_response_as_success(monkeypatch):
    monkeypatch.setattr(harvestguard_ai, "_call_gemini", lambda *args, **kwargs: {})
    with pytest.raises(AIError, match="did not complete"):
        check_connection("test-key")


def test_ui_stops_automatic_recovery_during_persistent_outage(monkeypatch):
    clock, calls = {"now": 1000}, []
    monkeypatch.setattr(harvestguard_ai_cache, "time", SimpleNamespace(time=lambda: clock["now"]))
    monkeypatch.setattr(harvestguard_ai_view, "time", SimpleNamespace(time=lambda: clock["now"]))
    monkeypatch.setattr(harvestguard_ai_view, "_setting", lambda name, default="": "test-key" if name == "GEMINI_API_KEY" else default)
    def generate(*args, **kwargs):
        calls.append(True)
        raise AIError("Google is temporarily busy.", code="provider_busy", retryable=True, http_status=503)
    monkeypatch.setattr(harvestguard_ai_view, "generate_analysis", generate)
    app = AppTest.from_file(str(ROOT / "harvestguard_app.py")).run(timeout=40)
    app.radio(key="page").set_value("Insights").run(timeout=40)
    assert not app.exception and len(calls) == 1
    clock["now"] += 31
    app.run(timeout=40)
    assert not app.exception and len(calls) == 2
    clock["now"] += 3600
    app.run(timeout=40)
    app.run(timeout=40)
    assert not app.exception and len(calls) == 2
    assert "Google is temporarily busy." in "\n".join(item.value for item in app.markdown)
    assert any("HTTP 503" in item.value for item in app.caption)
    app.button(key="retry_ai").click().run(timeout=40)
    assert not app.exception and len(calls) == 3


def test_filters_work_while_background_ai_is_waiting(monkeypatch):
    entered, release = Event(), Event()
    monkeypatch.setattr(harvestguard_ai_view, "_setting", lambda name, default="": "test-key" if name == "GEMINI_API_KEY" else default)
    def generate(packet, *args, **kwargs):
        entered.set()
        assert release.wait(timeout=20)
        return valid_response(packet)
    monkeypatch.setattr(harvestguard_ai_view, "generate_analysis", generate)
    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(harvestguard_ai_cache, "_executor", executor)
        try:
            app = AppTest.from_file(str(ROOT / "harvestguard_app.py")).run(timeout=10)
            app.radio(key="page").set_value("Insights").run(timeout=10)
            assert not app.exception
            assert entered.wait(timeout=5)
            assert "### Model findings" in "\n".join(item.value for item in app.markdown)
            app.selectbox(key="brief_district").select("Kamonyi").run(timeout=10)
            assert not app.exception
            assert "Kamonyi ranks" in "\n".join(item.value for item in app.markdown)
        finally:
            release.set()
        for future in list(harvestguard_ai_cache._jobs.values()):
            future.result(timeout=5)
        app.run(timeout=10)
        assert not app.exception
        assert any("Generated automatically" in item.value for item in app.caption)
