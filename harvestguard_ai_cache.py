"""Reuse validated explanations across refreshes and recover from temporary errors."""

import hashlib
import json
import logging
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from harvestguard_ai import AIError, generate_analysis, request_fingerprint, validate_analysis


CACHE_DIR = Path(__file__).resolve().parent / ".runtime" / "ai_cache"
_guard = threading.RLock()
_locks = {}
_failures = {}
_cooldowns = {}
_jobs = {}
_memory = {}
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="harvestguard-ai")
_logger = logging.getLogger(__name__)


def _read(path, packet, fingerprint, model):
    try:
        if path.stat().st_size > 1_000_000:
            return None
        saved = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(saved, dict) or saved.get("fingerprint") != fingerprint or saved.get("model") != model:
            return None
        if not isinstance(saved.get("generated_at"), str):
            return None
        saved["analysis"] = validate_analysis(saved["analysis"], packet)
        saved["cached"] = True
        return saved
    except (OSError, ValueError, KeyError, TypeError, AIError):
        return None


def _write(path, saved):
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(saved, handle, ensure_ascii=False, allow_nan=False)
        os.replace(temporary, path)
        return True
    except OSError:
        _logger.warning("AI explanation is available in memory, but could not be saved to disk.")
        return False
    finally:
        if temporary and temporary.exists():
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _cache_context(packet, api_key, model, question):
    fingerprint = request_fingerprint(packet, model, question)
    scope = hashlib.sha256((api_key + "\0" + model).encode("utf-8")).hexdigest()
    return fingerprint, scope, (scope, fingerprint), CACHE_DIR / scope / (fingerprint + ".json")


def lookup_explanation(packet, api_key, *, model, question):
    """Read saved work and cooldowns without sending a network request."""
    fingerprint, scope, cache_key, path = _cache_context(packet, api_key, model, question)
    saved = _read(path, packet, fingerprint, model)
    if saved:
        return saved
    now = time.time()
    with _guard:
        if cache_key in _memory:
            return {**_memory[cache_key], "cached": True}
        # Keep a selection's failure until a deliberate retry. Simply opening
        # metadata or changing another widget must not trigger another API call.
        if cache_key in _failures:
            return dict(_failures[cache_key])
        failure = _cooldowns.get(scope)
        if failure and failure["retry_at"] > now:
            return dict(failure)
        _cooldowns.pop(scope, None)
    return None


def retry_explanation(packet, api_key, *, model, question):
    """Release a failed selection after its cooldown, never duplicate live work."""
    _, scope, cache_key, _ = _cache_context(packet, api_key, model, question)
    with _guard:
        now = time.time()
        if any(failure and failure["retry_at"] > now for failure in
               (_failures.get(cache_key), _cooldowns.get(scope))):
            return False
        previous = _jobs.get(cache_key)
        if previous and not previous.done():
            return False
        _failures.pop(cache_key, None)
        _cooldowns.pop(scope, None)
        if previous and (previous.cancelled() or "analysis" not in previous.result()):
            _jobs.pop(cache_key, None)
        return True


def request_explanation(packet, api_key, *, model, question, generator=None):
    """Share a background task; Streamlit remains responsive while Google works."""
    _, _, cache_key, _ = _cache_context(packet, api_key, model, question)
    with _guard:
        previous = _jobs.get(cache_key)
        if previous and not previous.cancelled():
            return previous
        future = _executor.submit(get_explanation, packet, api_key, model=model, question=question, generator=generator)
        _jobs[cache_key] = future
        for key in list(_jobs):
            if len(_jobs) > 128 and key != cache_key and _jobs[key].done():
                del _jobs[key]
        return future


def get_explanation(packet, api_key, *, model, question, generator=None):
    """Return a verified answer or a recorded failure with a retry cooldown.

    Successful answers have no time expiry for these historical outputs. A change
    to the evidence, model, prompt or private key selects a different cache entry.
    No credentials are written to the cache or logs.
    """
    fingerprint, scope, cache_key, path = _cache_context(packet, api_key, model, question)
    saved = lookup_explanation(packet, api_key, model=model, question=question)
    if saved:
        return saved
    with _guard:
        # Serialize new requests for this private configuration. Cached answers
        # stay instant; quick selection changes cannot create overlapping calls.
        lock = _locks.setdefault(scope, threading.Lock())
    with lock:
        saved = lookup_explanation(packet, api_key, model=model, question=question)
        if saved:
            return saved
        try:
            analysis = (generator or generate_analysis)(packet, api_key, model=model, question=question)
            # Recheck before saving: only valid, matching explanations are reusable.
            analysis = validate_analysis(analysis, packet)
            saved = {
                "fingerprint": fingerprint, "analysis": analysis, "model": model,
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            if not _write(path, saved):
                with _guard:
                    _memory[cache_key] = saved
                    while len(_memory) > 64:
                        del _memory[next(iter(_memory))]
            return {**saved, "cached": False}
        except AIError as error:
            failure_error = error
        except Exception as error:
            # A background exception must not crash the page on future.result().
            # Log only the exception type; exception text can include credentials.
            _logger.error("Unexpected AI generation error: type=%s", type(error).__name__)
            failure_error = AIError("The AI explanation could not be completed. Your model results remain available. Retry the explanation.", code="internal")
        cooldown = failure_error.retry_after if failure_error.retry_after is not None else (30 if failure_error.retryable else 60)
        failure = {
            "error": str(failure_error), "code": failure_error.code, "retryable": failure_error.retryable,
            "http_status": failure_error.http_status, "retry_at": time.time() + max(1, cooldown),
        }
        if getattr(failure_error, "repair_hint", None):
            failure["validation_issue"] = failure_error.repair_hint
        with _guard:
            _failures[cache_key] = failure
            if failure_error.code in {"rate_limit", "daily_quota", "quota_unavailable", "access", "provider_busy", "connection", "timeout", "certificate", "network_blocked"}:
                _cooldowns[scope] = failure
        _logger.warning("AI explanation unavailable: category=%s http_status=%s selection=%s/%s/%s reason=%s", failure_error.code, failure_error.http_status, packet["identity"]["district"], packet["identity"]["crop"], packet["identity"]["season"], failure["error"])
        return failure
