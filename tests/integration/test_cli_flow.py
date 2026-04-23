from pathlib import Path
from subprocess import run

import pandas as pd

from judge_eval.config import config_hash, load_config, resolve_output_dir


def test_cli_flow(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment_name: integ",
                "datasets:",
                "  - name: evouna_tq",
                "    path: data/EVOUNA/TQ.json",
                "    sampling:",
                "      sample_size: 20",
                "      seed: 1",
                "filter:",
                "  improper: false",
                "judge_models:",
                "  - name: heuristic_dummy",
                "    provider: dummy",
                "    metadata:",
                "      dummy_strategy: heuristic",
                "output:",
                "  experiment_name: out",
                f"  base_dir: {tmp_path}",
                "telemetry:",
                "  enabled: false",
                "  provider: null",
            ]
        ),
        encoding="utf-8",
    )
    config, _ = load_config(config_path)
    output_dir = resolve_output_dir(config, config_hash(config_path))
    commands = [
        ["uv", "run", "judge-eval", "validate-config", str(config_path)],
        ["uv", "run", "judge-eval", "prepare-data", str(config_path)],
        ["uv", "run", "judge-eval", "run", str(config_path)],
        ["uv", "run", "judge-eval", "metrics", str(output_dir), "--bootstrap-iterations", "10"],
        ["uv", "run", "judge-eval", "report", str(output_dir)],
    ]
    for command in commands:
        completed = run(command, check=False, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr
    assert (output_dir / ".judge_eval_output.json").exists()
    assert (output_dir / "report.html").exists()
    assert (output_dir / "prompt_sensitivity.csv").exists()
    assert (output_dir / "reference_order_sensitivity.csv").exists()
    assert (output_dir / "dummy_answer_robustness.csv").exists()

    raw_before = (output_dir / "raw_predictions.jsonl").read_text(encoding="utf-8").splitlines()
    completed = run(
        ["uv", "run", "judge-eval", "run", str(config_path), "--resume"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    raw_after = (output_dir / "raw_predictions.jsonl").read_text(encoding="utf-8").splitlines()
    assert raw_before == raw_after

    parsed_before_retry = pd.read_parquet(output_dir / "parsed_predictions.parquet")
    first_row = parsed_before_retry.iloc[0].to_dict()
    injected_failure = {**first_row, "parse_status": "error", "error_message": "injected failure"}
    with (output_dir / "raw_predictions.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(pd.Series(injected_failure).to_json(force_ascii=False) + "\n")

    completed = run(
        ["uv", "run", "judge-eval", "retry-failures", str(config_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Retried predictions: 1 units" in completed.stdout

    raw_after_retry = (output_dir / "raw_predictions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(raw_after_retry) == len(raw_after) + 2

    parsed_after_retry = pd.read_parquet(output_dir / "parsed_predictions.parquet")
    repaired = parsed_after_retry.loc[parsed_after_retry["unit_key"].eq(first_row["unit_key"])].iloc[0]
    assert repaired["parse_status"] == "ok"
