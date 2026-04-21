from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import requests

from judge_eval.config import ModelConfig


@dataclass
class ProviderResponse:
    raw_output: str
    latency_ms: int
    input_tokens: int | None
    output_tokens: int | None
    estimated_cost: float | None


class ProviderError(RuntimeError):
    pass


def call_provider(model: ModelConfig, prompt: str) -> ProviderResponse:
    if model.provider == "dummy":
        return _call_dummy_provider(model, prompt)
    if model.provider in {"openai_compatible", "vllm"}:
        return _call_chat_completion_provider(model, prompt)
    if model.provider == "custom_http":
        return _call_custom_http_provider(model, prompt)
    if model.provider == "hf_local":
        return _call_hf_local_provider(model, prompt)
    raise ProviderError(f"unsupported provider: {model.provider}")


def _call_dummy_provider(model: ModelConfig, prompt: str) -> ProviderResponse:
    started = time.perf_counter()
    candidate_marker = "Candidate answer:\n"
    golden_marker = "Golden answer:\n"
    candidate = prompt.split(candidate_marker, 1)[1].split("\n\n", 1)[0].strip() if candidate_marker in prompt else ""
    golden = prompt.split(golden_marker, 1)[1].split("\n\n", 1)[0].strip() if golden_marker in prompt else ""
    provider_kind = model.metadata.get("dummy_strategy", model.name)
    if provider_kind == "perfect":
        label = candidate.lower() in golden.lower() or golden.lower() in candidate.lower()
    elif provider_kind == "always_true":
        label = True
    elif provider_kind == "always_false":
        label = False
    elif provider_kind == "invalid":
        raw = "reason first without valid json label maybe"
        return ProviderResponse(raw_output=raw, latency_ms=int((time.perf_counter() - started) * 1000), input_tokens=None, output_tokens=None, estimated_cost=0.0)
    else:
        label = _heuristic_label(golden, candidate)
    raw = json.dumps({"reason": f"Dummy strategy={provider_kind}", "label": label}, ensure_ascii=False)
    return ProviderResponse(
        raw_output=raw,
        latency_ms=int((time.perf_counter() - started) * 1000),
        input_tokens=len(prompt.split()),
        output_tokens=len(raw.split()),
        estimated_cost=0.0,
    )


def _heuristic_label(golden: str, candidate: str) -> bool:
    golden_norm = golden.lower()
    candidate_norm = candidate.lower()
    aliases = [item.strip() for item in golden_norm.split("/") if item.strip()]
    return any(alias in candidate_norm or candidate_norm in alias for alias in aliases)


def _call_chat_completion_provider(model: ModelConfig, prompt: str) -> ProviderResponse:
    if not model.endpoint:
        raise ProviderError(f"model {model.name} missing endpoint")
    started = time.perf_counter()
    headers = {"Content-Type": "application/json"}
    if model.api_key_env:
        import os

        api_key = os.environ.get(model.api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
    payload: dict[str, Any] = {
        "model": model.model or model.model_path or model.name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": model.temperature,
        "max_tokens": model.max_tokens,
    }
    response = requests.post(model.endpoint, headers=headers, json=payload, timeout=120)
    if response.status_code >= 400:
        raise ProviderError(f"provider call failed: {response.status_code} {response.text}")
    data = response.json()
    choice = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return ProviderResponse(
        raw_output=choice,
        latency_ms=int((time.perf_counter() - started) * 1000),
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
        estimated_cost=None,
    )


def _call_custom_http_provider(model: ModelConfig, prompt: str) -> ProviderResponse:
    if not model.endpoint:
        raise ProviderError(f"model {model.name} missing endpoint")
    started = time.perf_counter()
    response = requests.post(
        model.endpoint,
        json={"model": model.model or model.name, "prompt": prompt, "max_tokens": model.max_tokens},
        timeout=120,
    )
    if response.status_code >= 400:
        raise ProviderError(f"provider call failed: {response.status_code} {response.text}")
    data = response.json()
    content = data.get("content") or data.get("text") or data.get("output")
    if not isinstance(content, str):
        raise ProviderError("custom_http response missing string content")
    return ProviderResponse(
        raw_output=content,
        latency_ms=int((time.perf_counter() - started) * 1000),
        input_tokens=data.get("input_tokens"),
        output_tokens=data.get("output_tokens"),
        estimated_cost=data.get("estimated_cost"),
    )


def _call_hf_local_provider(model: ModelConfig, prompt: str) -> ProviderResponse:
    started = time.perf_counter()
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise ProviderError(
            "hf_local provider requires the optional 'transformers' dependency"
        ) from exc
    generator = pipeline(
        "text-generation",
        model=model.model_path or model.model,
    )
    outputs = generator(prompt, max_new_tokens=model.max_tokens, temperature=model.temperature)
    if not outputs or "generated_text" not in outputs[0]:
        raise ProviderError("hf_local provider did not return generated_text")
    generated_text = outputs[0]["generated_text"]
    raw_output = generated_text[len(prompt) :] if generated_text.startswith(prompt) else generated_text
    return ProviderResponse(
        raw_output=raw_output.strip(),
        latency_ms=int((time.perf_counter() - started) * 1000),
        input_tokens=None,
        output_tokens=None,
        estimated_cost=None,
    )
