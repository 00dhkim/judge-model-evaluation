from __future__ import annotations

import json
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from judge_eval.config import ExperimentConfig
from judge_eval.parsing import parse_model_output
from judge_eval.prompts import build_prompt
from judge_eval.providers import ProviderError, call_provider
from judge_eval.telemetry import TelemetryIds, TelemetrySession, json_io_attributes, telemetry_manifest, write_telemetry_manifest
from judge_eval.utils import ensure_dir, stable_hash


@dataclass
class RunOverrides:
    sample_size: int | None = None
    seed: int | None = None
    resume: bool = False


def prepare_output_dir(output_dir: str | Path) -> Path:
    return ensure_dir(Path(output_dir))


def write_resolved_config(path: Path, payload: dict[str, Any]) -> None:
    import yaml

    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def finalized_keys_from_parquet(path: Path) -> set[str]:
    if not path.exists():
        return set()
    frame = pd.read_parquet(path)
    return set(frame["unit_key"].tolist()) if "unit_key" in frame.columns else set()


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def run_predictions(
    config: ExperimentConfig,
    normalized_samples: pd.DataFrame,
    output_dir: Path,
    config_hash_value: str,
    dataset_hash: str,
    resolved_config: dict[str, Any],
    resume: bool = False,
) -> pd.DataFrame:
    raw_predictions_path = output_dir / "raw_predictions.jsonl"
    parsed_rows: list[dict[str, Any]] = []
    finished = finalized_keys_from_parquet(output_dir / "parsed_predictions.parquet") if resume else set()
    telemetry = TelemetrySession(
        config.telemetry.enabled,
        config.telemetry.project_name,
        str(output_dir),
        ax_space=config.telemetry.space,
        ax_profile=config.telemetry.profile,
        dataset_name=config.telemetry.dataset_name,
    )
    evaluation_rows = _build_evaluation_rows(config, normalized_samples)
    if not raw_predictions_path.exists():
        raw_predictions_path.touch()
    with raw_predictions_path.open("a", encoding="utf-8") as handle:
        try:
            for sample in evaluation_rows:
                for model in config.judge_models:
                    model_family = model.metadata.get("model_family", model.provider)
                    unit_key = stable_hash(
                        [sample["sample_id"], model.name, sample["prompt_template"], sample["variant_type"], sample["variant_id"]]
                    )
                    if resume and unit_key in finished:
                        continue
                    final_record = _run_single_attempt(
                        config=config,
                        sample=sample,
                        model_family=model_family,
                        model_name=model.name,
                        provider=model.provider,
                        prompt_template=sample["prompt_template"],
                        config_hash_value=config_hash_value,
                        dataset_hash=dataset_hash,
                        unit_key=unit_key,
                        raw_handle=handle,
                        telemetry=telemetry,
                    )
                    parsed_rows.append(final_record)
        finally:
            telemetry.shutdown()
    write_telemetry_manifest(output_dir / "telemetry_manifest.json", telemetry_manifest(telemetry))
    parsed = pd.DataFrame(parsed_rows)
    existing_path = output_dir / "parsed_predictions.parquet"
    if existing_path.exists():
        existing = pd.read_parquet(existing_path)
        if parsed.empty:
            return existing
        combined = pd.concat([existing, parsed], ignore_index=True)
        combined = combined.drop_duplicates(subset=["unit_key"], keep="last")
        combined.to_parquet(existing_path, index=False)
        return combined
    if not parsed.empty:
        parsed.to_parquet(existing_path, index=False)
    return parsed


def _is_yes_no_question(question: str) -> bool:
    normalized = question.strip().lower()
    prefixes = ("is ", "are ", "was ", "were ", "do ", "does ", "did ", "can ", "could ", "will ", "would ", "has ", "have ", "had ")
    return normalized.startswith(prefixes)


def _shuffle_aliases(aliases: list[str]) -> list[list[str]]:
    if len(aliases) <= 1:
        return []
    variants = [
        list(reversed(aliases)),
        aliases[1:] + aliases[:1],
        aliases[-1:] + aliases[:-1],
    ]
    unique: list[list[str]] = []
    seen = set()
    for variant in variants:
        key = tuple(variant)
        if key not in seen and key != tuple(aliases):
            seen.add(key)
            unique.append(variant)
    return unique


def _build_evaluation_rows(config: ExperimentConfig, normalized_samples: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base_prompt = config.evaluation.prompt_templates[0]
    for sample in normalized_samples.to_dict(orient="records"):
        base = dict(sample)
        base["variant_type"] = "base"
        base["variant_id"] = "base"
        base["variant_group"] = sample["sample_id"]
        base["parent_sample_id"] = sample["sample_id"]
        base["prompt_template"] = base_prompt
        base["variant_metadata"] = {}
        rows.append(base)
        if config.evaluation.enable_prompt_sensitivity:
            for prompt_template in config.evaluation.prompt_templates:
                variant = dict(sample)
                variant["variant_type"] = "prompt_sensitivity"
                variant["variant_id"] = prompt_template
                variant["variant_group"] = sample["sample_id"]
                variant["parent_sample_id"] = sample["sample_id"]
                variant["prompt_template"] = prompt_template
                variant["variant_metadata"] = {"prompt_template": prompt_template}
                rows.append(variant)
        if config.evaluation.enable_reference_order_sensitivity:
            for idx, aliases in enumerate(_shuffle_aliases(list(sample["golden_aliases"])), start=1):
                variant = dict(sample)
                variant["golden_aliases"] = aliases
                variant["golden_answer"] = "/".join(aliases)
                variant["variant_type"] = "reference_order"
                variant["variant_id"] = f"reference_order_{idx}"
                variant["variant_group"] = sample["sample_id"]
                variant["parent_sample_id"] = sample["sample_id"]
                variant["prompt_template"] = base_prompt
                variant["variant_metadata"] = {"alias_order": aliases}
                rows.append(variant)
        if config.evaluation.enable_dummy_answer_test:
            dummy_specs = [
                ("gold_answer_verbatim", list(sample["golden_aliases"])[0] if list(sample["golden_aliases"]) else sample["golden_answer"], True),
                ("yes", "Yes", True if _is_yes_no_question(sample["question"]) else False),
                ("sure", "Sure", False),
                ("repeat_question", sample["question"], False),
                ("empty", "", False),
            ]
            for dummy_name, candidate_answer, expected_label in dummy_specs:
                variant = dict(sample)
                variant["candidate_answer"] = candidate_answer
                variant["human_label"] = expected_label
                variant["variant_type"] = "dummy_answer"
                variant["variant_id"] = dummy_name
                variant["variant_group"] = sample["sample_id"]
                variant["parent_sample_id"] = sample["sample_id"]
                variant["prompt_template"] = base_prompt
                variant["variant_metadata"] = {
                    "dummy_class": dummy_name,
                    "yes_no_caution": dummy_name == "yes" and _is_yes_no_question(sample["question"]),
                }
                rows.append(variant)
    return rows


def _run_single_attempt(
    config: ExperimentConfig,
    sample: dict[str, Any],
    model_family: str,
    model_name: str,
    provider: str,
    prompt_template: str,
    config_hash_value: str,
    dataset_hash: str,
    unit_key: str,
    raw_handle: Any,
    telemetry: TelemetrySession,
) -> dict[str, Any]:
    retry_limit = max(config.evaluation.retry_count, 0)
    last_record: dict[str, Any] | None = None
    model_config = next(model for model in config.judge_models if model.name == model_name)
    for retry_count in range(retry_limit + 1):
        attempt_input = {
            "question": sample["question"],
            "golden_answer": sample["golden_answer"],
            "golden_aliases": list(sample["golden_aliases"]),
            "candidate_answer": sample["candidate_answer"],
            "human_label": sample["human_label"],
            "variant_type": sample["variant_type"],
            "variant_id": sample["variant_id"],
            "variant_metadata": sample["variant_metadata"],
            "prompt_template": prompt_template,
        }
        with telemetry.start_span(
            "eval.judge_sample",
            {
                "openinference.span.kind": "EVALUATOR",
                "arize.project.name": config.telemetry.project_name,
                "eval.name": "judge_eval",
                "eval.label": str(sample["human_label"]).lower(),
                "judge_eval.unit_key": unit_key,
                "judge_eval.sample_id": sample["sample_id"],
                "judge_eval.retry_count": retry_count,
                "judge_eval.variant_type": sample["variant_type"],
                "judge_eval.prompt_template": prompt_template,
                "judge_eval.judge_model": model_name,
                "judge_eval.provider": provider,
                **json_io_attributes("input", attempt_input),
            },
        ) as span_ids:
            try:
                prompt_inputs = {
                    "question": sample["question"],
                    "golden_answer": sample["golden_answer"],
                    "golden_aliases": list(sample["golden_aliases"]),
                    "candidate_answer": sample["candidate_answer"],
                    "template": prompt_template,
                }
                with telemetry.start_span(
                    "prompt.render",
                    {
                        "openinference.span.kind": "PROMPT",
                        "judge_eval.prompt_template": prompt_template,
                        **json_io_attributes("input", prompt_inputs),
                    },
                ) as prompt_span:
                    prompt = build_prompt(**prompt_inputs)
                    prompt_span.set_attributes({"output.value": prompt, "output.mime_type": "text/plain"})
                llm_attributes = {
                    "openinference.span.kind": "LLM",
                    "llm.model_name": model_config.model or model_config.model_path or model_config.name,
                    "llm.provider": provider,
                    "llm.invocation_parameters": json.dumps(
                        {
                            "temperature": model_config.temperature,
                            "max_tokens": model_config.max_tokens,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "llm.input_messages.0.message.role": "user",
                    "llm.input_messages.0.message.content": prompt,
                    "llm.prompt_template.template": prompt_template,
                    **json_io_attributes("input", {"prompt": prompt}),
                }
                with telemetry.start_span("llm.judge", llm_attributes) as llm_span:
                    try:
                        provider_response = call_provider(model_config, prompt)
                    except ProviderError as exc:
                        llm_span.set_attributes(
                            {
                                **json_io_attributes("output", {"error_message": str(exc)}),
                                "llm.error": str(exc),
                            }
                        )
                        raise
                    llm_span.set_attributes(
                        {
                            "output.value": provider_response.raw_output,
                            "output.mime_type": "text/plain",
                            "llm.output_messages.0.message.role": "assistant",
                            "llm.output_messages.0.message.content": provider_response.raw_output,
                            "llm.token_count.prompt": provider_response.input_tokens,
                            "llm.token_count.completion": provider_response.output_tokens,
                            "llm.cost.total": provider_response.estimated_cost,
                            "llm.latency_ms": provider_response.latency_ms,
                        }
                    )
                parsed = parse_model_output(provider_response.raw_output)
                parse_status = "ok" if retry_count == 0 and parsed["parse_method"] != "invalid" else "retry_ok"
                if parsed["parse_method"] == "invalid":
                    parse_status = "invalid"
                parse_output = {
                    "judge_reason": parsed["judge_reason"],
                    "parsed_label": parsed["parsed_label"],
                    "parse_method": parsed["parse_method"],
                    "parse_status": parse_status,
                }
                with telemetry.start_span(
                    "eval.parse_output",
                    {
                        "openinference.span.kind": "EVALUATOR",
                        "eval.name": "judge_eval.parse_output",
                        "eval.label": str(parsed["parsed_label"]).lower() if parsed["parsed_label"] is not None else "",
                        **json_io_attributes("input", {"raw_output": provider_response.raw_output}),
                        **json_io_attributes("output", parse_output),
                    },
                ):
                    pass
                record = _build_attempt_record(
                    config=config,
                    sample=sample,
                    model_family=model_family,
                    model_name=model_name,
                    provider=provider,
                    prompt_template=prompt_template,
                    config_hash_value=config_hash_value,
                    dataset_hash=dataset_hash,
                    unit_key=unit_key,
                    retry_count=retry_count,
                    span_ids=span_ids,
                    judge_reason=parsed["judge_reason"],
                    parsed_label=parsed["parsed_label"],
                    parse_status=parse_status,
                    raw_output=provider_response.raw_output,
                    latency_ms=provider_response.latency_ms,
                    input_tokens=provider_response.input_tokens,
                    output_tokens=provider_response.output_tokens,
                    estimated_cost=provider_response.estimated_cost,
                    error_message=None,
                )
                span_ids.set_attributes(
                    {
                        **json_io_attributes(
                            "output",
                            {
                                "judge_reason": parsed["judge_reason"],
                                "parsed_label": parsed["parsed_label"],
                                "parse_status": parse_status,
                                "error_message": None,
                            },
                        ),
                        "eval.score": 1.0 if parsed["parsed_label"] == sample["human_label"] else 0.0,
                        "eval.explanation": parsed["judge_reason"] or "",
                    }
                )
                raw_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                raw_handle.flush()
                last_record = record
                if parsed["parse_method"] != "invalid":
                    return record
                warnings.warn(f"Invalid output for unit {unit_key} on attempt {retry_count + 1}", stacklevel=2)
            except ProviderError as exc:
                record = _build_attempt_record(
                    config=config,
                    sample=sample,
                    model_family=model_family,
                    model_name=model_name,
                    provider=provider,
                    prompt_template=prompt_template,
                    config_hash_value=config_hash_value,
                    dataset_hash=dataset_hash,
                    unit_key=unit_key,
                    retry_count=retry_count,
                    span_ids=span_ids,
                    judge_reason=None,
                    parsed_label=None,
                    parse_status="error",
                    raw_output="",
                    latency_ms=None,
                    input_tokens=None,
                    output_tokens=None,
                    estimated_cost=None,
                    error_message=str(exc),
                )
                span_ids.set_attributes(
                    {
                        **json_io_attributes(
                            "output",
                            {
                                "judge_reason": None,
                                "parsed_label": None,
                                "parse_status": "error",
                                "error_message": str(exc),
                            },
                        ),
                        "eval.explanation": str(exc),
                    }
                )
                raw_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                raw_handle.flush()
                last_record = record
    return last_record or {}


def _build_attempt_record(
    *,
    config: ExperimentConfig,
    sample: dict[str, Any],
    model_family: str,
    model_name: str,
    provider: str,
    prompt_template: str,
    config_hash_value: str,
    dataset_hash: str,
    unit_key: str,
    retry_count: int,
    span_ids: TelemetryIds,
    judge_reason: str | None,
    parsed_label: bool | None,
    parse_status: str,
    raw_output: str,
    latency_ms: int | None,
    input_tokens: int | None,
    output_tokens: int | None,
    estimated_cost: float | None,
    error_message: str | None,
) -> dict[str, Any]:
    return {
        "experiment_name": config.experiment_name,
        "sample_id": sample["sample_id"],
        "parent_sample_id": sample["parent_sample_id"],
        "judge_model": model_name,
        "model_family": model_family,
        "provider": provider,
        "prompt_template": prompt_template,
        "prompt_template_version": "v1",
        "variant_type": sample["variant_type"],
        "variant_id": sample["variant_id"],
        "variant_group": sample["variant_group"],
        "seed": 0,
        "judge_reason": judge_reason,
        "parsed_label": parsed_label,
        "parse_status": parse_status,
        "raw_output": raw_output,
        "error_message": error_message,
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost": estimated_cost,
        "retry_count": retry_count,
        "trace_id": span_ids.trace_id,
        "span_id": span_ids.span_id,
        "git_commit": git_commit(),
        "config_hash": config_hash_value,
        "dataset_hash": dataset_hash,
        "unit_key": unit_key,
        "variant_metadata": sample["variant_metadata"],
        "human_label": sample["human_label"],
        "dataset": sample["dataset"],
        "answer_source": sample["answer_source"],
        "answer_length_bucket": sample["answer_length_bucket"],
        "golden_answer_alias_count": sample["golden_answer_alias_count"],
    }
