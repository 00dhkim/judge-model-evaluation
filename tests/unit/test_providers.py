from types import SimpleNamespace

import pytest
import requests

from judge_eval.config import ModelConfig
from judge_eval.providers import (
    ProviderError,
    _RATE_LIMIT_MAX_RETRIES,
    _call_chat_completion_provider,
    estimate_openai_text_cost,
)


def _model() -> ModelConfig:
    return ModelConfig(
        name="gemma",
        provider="openai_compatible",
        model="gemma-3-12b-it",
        endpoint="https://example.com/v1/chat/completions",
    )


def _openai_model() -> ModelConfig:
    return ModelConfig(
        name="gpt54mini",
        provider="openai_compatible",
        model="gpt-5.4-mini",
        endpoint="https://api.openai.com/v1/chat/completions",
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


def test_chat_completion_uses_max_completion_tokens_for_openai_endpoint(monkeypatch):
    payloads: list[dict] = []

    def fake_post(*args, **kwargs):
        payloads.append(dict(kwargs["json"]))
        return SimpleNamespace(
            status_code=200,
            json=lambda: {
                "choices": [{"message": {"content": '{"label": true, "reason": "ok"}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    monkeypatch.setattr("judge_eval.providers.requests.post", fake_post)

    response = _call_chat_completion_provider(_openai_model(), "prompt")

    assert response.raw_output == '{"label": true, "reason": "ok"}'
    assert len(payloads) == 1
    assert "max_tokens" not in payloads[0]
    assert payloads[0]["max_completion_tokens"] == 256


def test_chat_completion_uses_max_tokens_for_non_openai_endpoint(monkeypatch):
    payloads: list[dict] = []

    def fake_post(*args, **kwargs):
        payloads.append(dict(kwargs["json"]))
        return SimpleNamespace(
            status_code=200,
            json=lambda: {
                "choices": [{"message": {"content": '{"label": true, "reason": "ok"}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    monkeypatch.setattr("judge_eval.providers.requests.post", fake_post)

    response = _call_chat_completion_provider(_model(), "prompt")

    assert response.raw_output == '{"label": true, "reason": "ok"}'
    assert len(payloads) == 1
    assert payloads[0]["max_tokens"] == 256
    assert "max_completion_tokens" not in payloads[0]


def test_estimate_openai_text_cost_uses_cached_and_output_pricing():
    cost = estimate_openai_text_cost(
        "gpt-5.4-mini",
        {
            "prompt_tokens": 1_000_000,
            "completion_tokens": 1_000_000,
            "prompt_tokens_details": {"cached_tokens": 250_000},
        },
    )

    assert cost == 5.08125


def test_chat_completion_sets_estimated_cost_for_openai_endpoint(monkeypatch):
    def fake_post(*args, **kwargs):
        return SimpleNamespace(
            status_code=200,
            json=lambda: {
                "choices": [{"message": {"content": '{"label": true, "reason": "ok"}'}}],
                "usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 200,
                    "prompt_tokens_details": {"cached_tokens": 100},
                },
            },
        )

    monkeypatch.setattr("judge_eval.providers.requests.post", fake_post)

    response = _call_chat_completion_provider(_openai_model(), "prompt")

    assert response.estimated_cost == pytest.approx(0.0015825)
