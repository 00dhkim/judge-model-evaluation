from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, computed_field, model_validator

from judge_eval.settings import ARIZE_ENV_VARS, PROMPT_TEMPLATES
from judge_eval.utils import stable_hash


class SamplingConfig(BaseModel):
    sample_size: int | None = None
    seed: int | None = None

    @model_validator(mode="after")
    def validate_seed_requires_size(self) -> "SamplingConfig":
        if self.seed is not None and self.sample_size is None:
            raise ValueError("sampling.seed requires sampling.sample_size to be set")
        return self


class DatasetConfig(BaseModel):
    name: str
    path: str
    sampling: SamplingConfig | None = None


class FilterConfig(BaseModel):
    improper: bool = False
    exclude_non_boolean_labels: bool = True
    exclude_empty_candidate_answers: bool = True


class ModelConfig(BaseModel):
    name: str
    provider: Literal["dummy", "openai_compatible", "vllm", "hf_local", "custom_http"]
    model: str | None = None
    model_path: str | None = None
    endpoint: str | None = None
    api_key_env: str | None = None
    temperature: float = 0.0
    max_tokens: int = 256
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_provider_requirements(self) -> "ModelConfig":
        if self.provider == "dummy":
            return self
        if self.provider in {"openai_compatible", "vllm"} and not self.model:
            raise ValueError(f"provider '{self.provider}' requires model")
        if self.provider in {"openai_compatible", "vllm", "custom_http"} and not self.endpoint:
            raise ValueError(f"provider '{self.provider}' requires endpoint")
        if self.provider == "hf_local" and not (self.model or self.model_path):
            raise ValueError("provider 'hf_local' requires model or model_path")
        return self


class EvaluationConfig(BaseModel):
    prompt_templates: list[Literal["minimal", "guideline", "guideline_with_examples"]] = Field(
        default_factory=lambda: ["minimal", "guideline", "guideline_with_examples"]
    )
    retry_count: int = 1
    invalid_output_policy: Literal["store_invalid"] = "store_invalid"
    bootstrap_iterations: int = 200
    enable_prompt_sensitivity: bool = True
    enable_reference_order_sensitivity: bool = True
    enable_dummy_answer_test: bool = True

    @model_validator(mode="after")
    def validate_templates(self) -> "EvaluationConfig":
        unknown = set(self.prompt_templates) - set(PROMPT_TEMPLATES)
        if unknown:
            raise ValueError(f"unsupported prompt templates: {sorted(unknown)}")
        return self


class OutputConfig(BaseModel):
    experiment_name: str
    base_dir: str = "outputs"
    save_raw_predictions: bool = True
    save_report: bool = True

    @computed_field
    @property
    def dir(self) -> str:
        date = datetime.now().strftime("%Y%m%d")
        return f"{self.base_dir}/{date}_{self.experiment_name}"


class TelemetryConfig(BaseModel):
    enabled: bool = False
    provider: Literal["arize"] | None = None
    project_name: str = "meta-judge-eval"
    dataset_name: str = "meta-judge-eval"
    space: str | None = None
    profile: str | None = None

    @model_validator(mode="after")
    def validate_provider(self) -> "TelemetryConfig":
        if self.enabled and self.provider != "arize":
            raise ValueError("telemetry.enabled requires telemetry.provider=arize")
        return self


class ExperimentConfig(BaseModel):
    experiment_name: str
    datasets: list[DatasetConfig]
    filter: FilterConfig = Field(default_factory=FilterConfig)
    judge_models: list[ModelConfig]
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    output: OutputConfig
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)

    @model_validator(mode="after")
    def validate_experiment(self) -> "ExperimentConfig":
        if not self.datasets:
            raise ValueError("at least one dataset is required")
        if not self.judge_models:
            raise ValueError("at least one judge model is required")
        return self


def _resolve_env_value(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_name = value[2:-1]
        return os.environ.get(env_name)
    return value


def _resolve_data(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: _resolve_data(_resolve_env_value(value)) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_resolve_data(item) for item in payload]
    return payload


def load_config(path: str | Path) -> tuple[ExperimentConfig, dict[str, Any]]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    resolved = _resolve_data(raw)
    config = ExperimentConfig.model_validate(resolved)
    _validate_telemetry_env(config)
    return config, resolved


def _validate_telemetry_env(config: ExperimentConfig) -> None:
    if config.telemetry.enabled:
        missing = [name for name in ARIZE_ENV_VARS if not os.environ.get(name)]
        if missing:
            raise ValueError(f"telemetry enabled but missing env vars: {', '.join(missing)}")


def resolved_config_with_redactions(path: str | Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    resolved = _resolve_data(raw)
    for model in resolved.get("judge_models", []):
        api_key_env = model.get("api_key_env")
        if api_key_env:
            model["api_key"] = "<redacted-env>"
    if resolved.get("telemetry", {}).get("enabled"):
        resolved["telemetry"]["env_refs"] = list(ARIZE_ENV_VARS)
    return resolved


def config_hash(path: str | Path) -> str:
    resolved = resolved_config_with_redactions(path)
    return stable_hash(resolved)


def validate_config_file(path: str | Path) -> list[str]:
    errors: list[str] = []
    try:
        load_config(path)
    except (ValidationError, ValueError) as exc:
        if isinstance(exc, ValidationError):
            errors.extend(error["msg"] for error in exc.errors())
        else:
            errors.append(str(exc))
    return errors
