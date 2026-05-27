from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "aggregate_optim_results.py"
SPEC = importlib.util.spec_from_file_location("aggregate_optim_results", SCRIPT_PATH)
assert SPEC is not None
aggregate_optim_results = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(aggregate_optim_results)


def test_aggregate_root_scans_all_prediction_files(tmp_path):
    root = tmp_path / "optim_final"
    method_a = root / "method_a"
    method_b = root / "method_b"
    method_a.mkdir(parents=True)
    method_b.mkdir(parents=True)

    pd.DataFrame(
        [
            {"sample_id": "s1", "parsed_label": True, "n_calls": 1, "total_cost": 0.1, "total_latency_ms": 10},
            {"sample_id": "s2", "parsed_label": False, "n_calls": 1, "total_cost": 0.2, "total_latency_ms": 20},
        ]
    ).to_json(method_a / "predictions.jsonl", orient="records", lines=True)
    pd.DataFrame(
        [
            {"sample_id": "s1", "parsed_label": False, "n_calls": 3, "total_cost": 0.3, "total_latency_ms": 30},
            {"sample_id": "s2", "parsed_label": False, "n_calls": 3, "total_cost": 0.4, "total_latency_ms": 40},
        ]
    ).to_json(method_b / "predictions.jsonl", orient="records", lines=True)
    sample_source = tmp_path / "normalized_samples.parquet"
    pd.DataFrame(
        [
            {"sample_id": "s1", "human_label": True},
            {"sample_id": "s2", "human_label": False},
        ]
    ).to_parquet(sample_source, index=False)

    summary, manifest = aggregate_optim_results.aggregate_root(
        root,
        judge="solar",
        sample_source=sample_source,
        generated_at="2026-05-27T00:00:00+00:00",
        commit="test-commit",
    )

    assert summary["method"].tolist() == ["method_a", "method_b"]
    assert summary["sample_count"].tolist() == [2, 2]
    assert summary["total_calls"].tolist() == [2, 6]
    assert summary.loc[summary["method"] == "method_a", "scotts_pi"].item() == 1.0
    assert (root / "summary_full.csv").exists()
    assert (root / "summary_manifest.json").exists()
    assert [item["method"] for item in manifest["files"]] == ["method_a", "method_b"]
