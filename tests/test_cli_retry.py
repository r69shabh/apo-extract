import asyncio

import litellm
import pytest

from apo_extract.cli import _install_429_retry


@pytest.fixture(autouse=True)
def _fast_sleep(monkeypatch):
    real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda *_a, **_k: real_sleep(0))


@pytest.fixture(autouse=True)
def _restore_acompletion():
    original = litellm.acompletion
    yield
    litellm.acompletion = original


def test_retry_wrapper_retries_429_and_pins_timeout():
    calls = {"n": 0}

    async def fake(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise Exception("Error code: 429 - resource exhausted, retry in 0.1s")
        return {"ok": True, **kwargs}

    litellm.acompletion = fake
    _install_429_retry(max_attempts=3)
    result = asyncio.run(litellm.acompletion(model="gemini/x", messages=[]))
    assert result["ok"] is True
    assert calls["n"] == 2
    assert litellm.request_timeout == 180
    assert result["timeout"] == 180


def test_retry_wrapper_retries_timeout_errors():
    calls = {"n": 0}

    async def fake(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 2:
            raise TimeoutError("Connection timed out after 180s")
        return {"ok": True}

    litellm.acompletion = fake
    _install_429_retry(max_attempts=3)
    result = asyncio.run(litellm.acompletion(model="gemini/x", messages=[]))
    assert result["ok"] is True
    assert calls["n"] == 2


def test_retry_wrapper_bounds_hard_timeouts():
    calls = {"n": 0}
    never = asyncio.Event()

    async def fake(*args, **kwargs):
        calls["n"] += 1
        await never.wait()

    litellm.acompletion = fake
    _install_429_retry(max_attempts=5, call_timeout=0.05)
    with pytest.raises(TimeoutError):
        asyncio.run(litellm.acompletion(model="gemini/x", messages=[]))
    assert calls["n"] == 3


def test_retry_wrapper_aborts_on_daily_quota_delay():
    calls = {"n": 0}

    async def fake(*args, **kwargs):
        calls["n"] += 1
        raise Exception(
            'Error code: 429 - RESOURCE_EXHAUSTED, limit: 20, model: gemini-3.8-flash, '
            '"retryDelay": "23646s"'
        )

    litellm.acompletion = fake
    _install_429_retry(max_attempts=5)
    with pytest.raises(RuntimeError, match="quota"):
        asyncio.run(litellm.acompletion(model="gemini/x", messages=[]))
    assert calls["n"] == 1
