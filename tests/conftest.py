"""Keep test AI answers and cooldowns separate from the real demo cache."""
import pytest
from concurrent.futures import Future

import harvestguard_ai_cache


class ImmediateExecutor:
    """Resolve background work deterministically in UI tests."""
    def submit(self, function, *args, **kwargs):
        future = Future()
        try:
            future.set_result(function(*args, **kwargs))
        except Exception as error:
            future.set_exception(error)
        return future


@pytest.fixture(autouse=True)
def isolated_ai_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(harvestguard_ai_cache, "CACHE_DIR", tmp_path / "ai_cache")
    monkeypatch.setattr(harvestguard_ai_cache, "_failures", {})
    monkeypatch.setattr(harvestguard_ai_cache, "_cooldowns", {})
    monkeypatch.setattr(harvestguard_ai_cache, "_locks", {})
    monkeypatch.setattr(harvestguard_ai_cache, "_jobs", {})
    monkeypatch.setattr(harvestguard_ai_cache, "_memory", {})
    monkeypatch.setattr(harvestguard_ai_cache, "_executor", ImmediateExecutor())
