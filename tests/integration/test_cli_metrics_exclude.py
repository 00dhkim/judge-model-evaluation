from pathlib import Path
from subprocess import run

import pandas as pd


def _prediction_row(judge_model: str, parsed_label: bool, human_label: bool) -> dict[str, object]:
    return {
        "experiment_name": "exclude_case",
        "judge_model": judge_model,
        "prompt_template": "minimal",
        "variant_type": "base",
        "variant_group": "sample-1",
        "parsed_label": parsed_label,
        "human_label": human_label,
        "latency_ms": 1,
        "estimated_cost": 0,
        "dataset": "TQ",
        "answer_source": "fid",
        "answer_length_bucket": "short",
        "golden_answer_alias_count": 1,
        "model_family": "dummy",
    }


def test_metrics_exclude_models_filters_metrics_output(tmp_path: Path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    pd.DataFrame(
        [
            _prediction_row("judge-a", True, True),
            _prediction_row("judge-b", True, False),
        ]
    ).to_parquet(output_dir / "parsed_predictions.parquet", index=False)
    (output_dir / "config.resolved.yaml").write_text("telemetry:\n  enabled: false\n", encoding="utf-8")

    completed = run(
        [
            "uv",
            "run",
            "judge-eval",
            "metrics",
            str(output_dir),
            "--bootstrap-iterations",
            "1",
            "--exclude-models",
            "judge-b",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    metrics = pd.read_csv(output_dir / "metrics_overall.csv")
    assert metrics["judge_model"].tolist() == ["judge-a"]
