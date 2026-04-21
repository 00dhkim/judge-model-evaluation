from datetime import datetime
from pathlib import Path
from subprocess import run


def test_cli_flow(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / f"{datetime.now().strftime('%Y%m%d')}_out"
    config_path.write_text(
        "\n".join(
            [
                "experiment_name: integ",
                "datasets:",
                "  - name: evouna_tq",
                "    path: data/EVOUNA/TQ.json",
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
    commands = [
        ["uv", "run", "judge-eval", "validate-config", str(config_path)],
        ["uv", "run", "judge-eval", "prepare-data", str(config_path), "--sample-size", "20", "--seed", "1"],
        ["uv", "run", "judge-eval", "run", str(config_path), "--sample-size", "20", "--seed", "1"],
        ["uv", "run", "judge-eval", "metrics", str(output_dir), "--bootstrap-iterations", "10"],
        ["uv", "run", "judge-eval", "report", str(output_dir)],
    ]
    for command in commands:
        completed = run(command, check=False, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr
    assert (output_dir / "report.md").exists()
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
