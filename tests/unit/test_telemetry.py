from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

from judge_eval.telemetry import TelemetrySession, build_metrics_dataset_file, sync_ax_dataset, telemetry_manifest


class DummyExporter:
    exported: list[dict[str, object]] = []

    def __init__(self, *args, **kwargs):
        pass

    def export(self, spans):
        for span in spans:
            DummyExporter.exported.append(
                {
                    "name": span.name,
                    "attributes": dict(span.attributes),
                    "parent": f"{span.parent.span_id:016x}" if span.parent is not None else None,
                }
            )
        return None

    def shutdown(self):
        return None

    def force_flush(self, timeout_millis=30000):
        return True


def _completed(returncode: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["ax"], returncode=returncode, stdout=stdout, stderr=stderr)


def test_telemetry_session_enabled(monkeypatch):
    DummyExporter.exported = []
    monkeypatch.setenv("ARIZE_API_KEY", "key")
    monkeypatch.setenv("ARIZE_SPACE_ID", "space")
    monkeypatch.setattr("judge_eval.telemetry.OTLPSpanExporter", DummyExporter)
    monkeypatch.setattr("judge_eval.telemetry._run_ax", lambda args, profile=None: _completed(0, stdout='{"project":"ok"}'))
    session = TelemetrySession(True, "meta-judge-eval")
    with session.start_span("eval.judge_sample", {"openinference.span.kind": "EVALUATOR"}) as root:
        root.set_attributes({"input.value": '{"question":"q"}', "input.mime_type": "application/json"})
        with session.start_span("llm.judge", {"openinference.span.kind": "LLM"}) as ids:
            ids.set_attributes({"output.value": "{}", "output.mime_type": "application/json"})
        assert ids.trace_id
        assert ids.span_id
    session.shutdown()
    manifest = telemetry_manifest(session)
    assert manifest["enabled"] is True
    assert manifest["project_name"] == "meta-judge-eval"
    assert manifest["dataset_name"] == "meta-judge-eval"
    assert manifest["ax"]["project"]["status"] == "exists"
    exported_names = [item["name"] for item in DummyExporter.exported]
    assert "eval.judge_sample" in exported_names
    assert "llm.judge" in exported_names
    llm_span = next(item for item in DummyExporter.exported if item["name"] == "llm.judge")
    assert llm_span["attributes"]["openinference.span.kind"] == "LLM"
    assert llm_span["attributes"]["output.value"] == "{}"


def test_build_metrics_dataset_file(tmp_path: Path):
    metrics_path = tmp_path / "metrics_overall.csv"
    parsed_path = tmp_path / "parsed_predictions.parquet"
    pd.DataFrame(
        [
            {
                "judge_model": "heuristic_dummy",
                "prompt_template": "minimal",
                "scotts_pi": 1.0,
            }
        ]
    ).to_csv(metrics_path, index=False)
    pd.DataFrame([{"experiment_name": "integ"}]).to_parquet(parsed_path, index=False)
    upload_path = build_metrics_dataset_file(
        metrics_path,
        parsed_path,
        tmp_path,
        project_name="meta-judge-eval",
        dataset_name="meta-judge-eval",
    )
    upload = pd.read_parquet(upload_path)
    assert upload.loc[0, "experiment_name"] == "integ"
    assert upload.loc[0, "telemetry_project_name"] == "meta-judge-eval"
    assert upload.loc[0, "telemetry_dataset_name"] == "meta-judge-eval"


def test_sync_ax_dataset_appends_when_dataset_exists(monkeypatch, tmp_path: Path):
    calls: list[list[str]] = []

    def fake_run(args, profile=None):
        calls.append(args)
        if args[:3] == ["datasets", "get", "meta-judge-eval"]:
            return _completed(0, stdout='{"dataset":"ok"}')
        return _completed(0, stdout='{"append":"ok"}')

    monkeypatch.setenv("ARIZE_SPACE_ID", "space")
    monkeypatch.setattr("judge_eval.telemetry._run_ax", fake_run)
    data_path = tmp_path / "metrics.parquet"
    pd.DataFrame([{"judge_model": "x"}]).to_parquet(data_path, index=False)
    status = sync_ax_dataset("meta-judge-eval", data_path)
    assert status["status"] == "appended"
    assert any(call[:3] == ["datasets", "append", "meta-judge-eval"] for call in calls)
