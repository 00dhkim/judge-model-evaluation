from types import SimpleNamespace

import requests

from judge_eval.config import ModelConfig
from judge_eval.providers import ProviderError, _RATE_LIMIT_MAX_RETRIES, _call_chat_completion_provider


def _model() -> ModelConfig:
    return ModelConfig(
        name="gemma",
        provider="openai_compatible",
        model="gemma-3-12b-it",
        endpoint="https://example.com/v1/chat/completions",
    )


def test_chat_completion_retries_timeouts_with_exponential_backoff(monkeypatch):
    calls = {"count": 0}
    sleeps: list[float] = []

    def fake_post(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] < 3:
            raise requests.exceptions.ReadTimeout("timed out")
        return SimpleNamespace(
            status_code=200,
            json=lambda: {
                "choices": [{"message": {"content": '{"label": true, "reason": "ok"}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    monkeypatch.setattr("judge_eval.providers.requests.post", fake_post)
    monkeypatch.setattr("judge_eval.providers.time.sleep", lambda delay: sleeps.append(delay))

    response = _call_chat_completion_provider(_model(), "prompt")

    assert response.raw_output == '{"label": true, "reason": "ok"}'
    assert calls["count"] == 3
    assert sleeps == [1.0, 2.0]


def test_chat_completion_raises_after_exhausting_timeout_retries(monkeypatch):
    sleeps: list[float] = []

    def raise_timeout(*args, **kwargs):
        raise requests.exceptions.ReadTimeout("timed out")

    monkeypatch.setattr("judge_eval.providers.requests.post", raise_timeout)
    monkeypatch.setattr("judge_eval.providers.time.sleep", lambda delay: sleeps.append(delay))

    try:
        _call_chat_completion_provider(_model(), "prompt")
    except ProviderError as exc:
        assert "provider request failed" in str(exc)
        assert "timed out" in str(exc)
    else:
        raise AssertionError("ProviderError was not raised")

    assert len(sleeps) == _RATE_LIMIT_MAX_RETRIES
    assert sleeps[:3] == [1.0, 2.0, 4.0]
