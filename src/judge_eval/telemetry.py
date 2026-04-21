from __future__ import annotations

import json
import os
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


@dataclass
class TelemetryIds:
    trace_id: str
    span_id: str
    _span: Any | None = None

    def set_attribute(self, key: str, value: Any) -> None:
        if self._span is not None and value is not None:
            self._span.set_attribute(key, value)

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        for key, value in attributes.items():
            self.set_attribute(key, value)


def json_value(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def json_io_attributes(prefix: str, payload: Any) -> dict[str, str]:
    return {
        f"{prefix}.value": json_value(payload),
        f"{prefix}.mime_type": "application/json",
    }


def _resolve_ax_space(explicit_space: str | None) -> str:
    return explicit_space or os.environ["ARIZE_SPACE_ID"]


def _run_ax(args: list[str], profile: str | None = None) -> subprocess.CompletedProcess[str]:
    command = ["ax", *args]
    if profile:
        command.extend(["--profile", profile])
    return subprocess.run(command, check=False, capture_output=True, text=True)


def _status_payload(action: str, result: subprocess.CompletedProcess[str], *, created: bool = False) -> dict[str, object]:
    payload: dict[str, object] = {
        "action": action,
        "created": created,
        "returncode": result.returncode,
    }
    if result.stdout.strip():
        payload["stdout"] = result.stdout.strip()
    if result.stderr.strip():
        payload["stderr"] = result.stderr.strip()
    return payload


def ensure_ax_project(project_name: str, *, space: str | None = None, profile: str | None = None) -> dict[str, object]:
    resolved_space = _resolve_ax_space(space)
    lookup = _run_ax(["projects", "get", project_name, "--space", resolved_space, "--output", "json"], profile=profile)
    if lookup.returncode == 0:
        return {
            "enabled": True,
            "space": resolved_space,
            "project_name": project_name,
            "status": "exists",
            "lookup": _status_payload("get", lookup),
        }
    created = _run_ax(["projects", "create", "--name", project_name, "--space", resolved_space, "--output", "json"], profile=profile)
    status = "created" if created.returncode == 0 else "error"
    return {
        "enabled": True,
        "space": resolved_space,
        "project_name": project_name,
        "status": status,
        "lookup": _status_payload("get", lookup),
        "create": _status_payload("create", created, created=status == "created"),
    }


def sync_ax_dataset(
    dataset_name: str,
    dataset_file: str | Path,
    *,
    space: str | None = None,
    profile: str | None = None,
) -> dict[str, object]:
    resolved_space = _resolve_ax_space(space)
    dataset_path = str(dataset_file)
    lookup = _run_ax(["datasets", "get", dataset_name, "--space", resolved_space, "--output", "json"], profile=profile)
    if lookup.returncode == 0:
        append = _run_ax(
            ["datasets", "append", dataset_name, "--space", resolved_space, "--file", dataset_path, "--output", "json"],
            profile=profile,
        )
        return {
            "enabled": True,
            "space": resolved_space,
            "dataset_name": dataset_name,
            "status": "appended" if append.returncode == 0 else "error",
            "lookup": _status_payload("get", lookup),
            "append": _status_payload("append", append),
        }
    created = _run_ax(
        ["datasets", "create", "--name", dataset_name, "--space", resolved_space, "--file", dataset_path, "--output", "json"],
        profile=profile,
    )
    status = "created" if created.returncode == 0 else "error"
    return {
        "enabled": True,
        "space": resolved_space,
        "dataset_name": dataset_name,
        "status": status,
        "lookup": _status_payload("get", lookup),
        "create": _status_payload("create", created, created=status == "created"),
    }


def build_metrics_dataset_file(
    metrics_overall: str | Path,
    parsed_predictions: str | Path,
    output_dir: str | Path,
    *,
    project_name: str,
    dataset_name: str,
) -> Path:
    import pandas as pd

    metrics = pd.read_csv(metrics_overall)
    parsed = pd.read_parquet(parsed_predictions)
    experiment_name = str(parsed["experiment_name"].iloc[0]) if not parsed.empty else ""
    upload = metrics.copy()
    upload.insert(0, "experiment_name", experiment_name)
    upload.insert(1, "telemetry_project_name", project_name)
    upload.insert(2, "telemetry_dataset_name", dataset_name)
    upload["source_output_dir"] = str(Path(output_dir))
    upload_path = Path(output_dir) / "arize_metrics_dataset.parquet"
    upload.to_parquet(upload_path, index=False)
    return upload_path


class TelemetrySession:
    def __init__(
        self,
        enabled: bool,
        project_name: str,
        output_dir: str | None = None,
        *,
        ax_space: str | None = None,
        ax_profile: str | None = None,
        dataset_name: str | None = None,
    ) -> None:
        self.enabled = enabled
        self.project_name = project_name
        self.dataset_name = dataset_name or project_name
        self.output_dir = output_dir
        self.ax_space = ax_space
        self.ax_profile = ax_profile
        self.endpoint = os.environ.get("ARIZE_OTLP_ENDPOINT", "https://otlp.arize.com/v1")
        self.provider: TracerProvider | None = None
        self.tracer = None
        self.ax_project_status: dict[str, object] = {"enabled": False, "status": "disabled"}
        if enabled:
            self.ax_project_status = ensure_ax_project(project_name, space=ax_space, profile=ax_profile)
            if self.ax_project_status.get("status") == "error":
                raise RuntimeError(f"failed to ensure Arize project via ax: {project_name}")
            resource = Resource.create(
                {
                    "service.name": "judge-eval",
                    "service.version": "0.1.0",
                    "openinference.project.name": project_name,
                    "arize.project.name": project_name,
                }
            )
            provider = TracerProvider(resource=resource)
            exporter = OTLPSpanExporter(
                endpoint=self.endpoint,
                headers=(("space_id", os.environ["ARIZE_SPACE_ID"]), ("api_key", os.environ["ARIZE_API_KEY"])),
            )
            provider.add_span_processor(BatchSpanProcessor(exporter))
            self.provider = provider
            self.tracer = provider.get_tracer("judge_eval")

    @contextmanager
    def start_span(self, name: str, attributes: dict[str, object]) -> Iterator[TelemetryIds]:
        if not self.enabled or self.tracer is None:
            yield TelemetryIds(trace_id="", span_id="")
            return
        span_attributes = dict(attributes)
        if self.enabled:
            span_attributes.setdefault("arize.project.name", self.project_name)
            span_attributes.setdefault("openinference.project.name", self.project_name)
        with self.tracer.start_as_current_span(name, attributes=span_attributes) as span:
            context = span.get_span_context()
            yield TelemetryIds(trace_id=f"{context.trace_id:032x}", span_id=f"{context.span_id:016x}", _span=span)

    def shutdown(self) -> None:
        if self.provider is not None:
            self.provider.shutdown()


def telemetry_manifest(
    session: TelemetrySession,
    dataset_status: dict[str, object] | None = None,
) -> dict[str, object]:
    env_refs = [name for name in ("ARIZE_API_KEY", "ARIZE_SPACE_ID") if os.environ.get(name)]
    manifest: dict[str, Any] = {
        "enabled": session.enabled,
        "provider": "arize" if session.enabled else None,
        "endpoint": session.endpoint if session.enabled else None,
        "project_name": session.project_name,
        "dataset_name": session.dataset_name,
        "space": session.ax_project_status.get("space") if session.enabled else None,
        "profile": session.ax_profile,
        "env_refs_present": env_refs,
        "ax": {
            "project": session.ax_project_status,
        },
        "trace_naming": {
            "root": "eval.judge_sample",
            "prompt_render": "prompt.render",
            "llm_judge": "llm.judge",
            "parse_output": "eval.parse_output",
        },
    }
    if dataset_status is not None:
        manifest["ax"]["dataset"] = dataset_status
    return manifest


def load_telemetry_manifest(path: str | Path) -> dict[str, object]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def write_telemetry_manifest(path: str | Path, payload: dict[str, object]) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
