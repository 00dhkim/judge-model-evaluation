from pathlib import Path
from subprocess import run


def test_report_contains_required_sections(tmp_path: Path):
    config = tmp_path / "config.yaml"
    out = tmp_path / "out"
    config.write_text(
        "\n".join(
            [
                "experiment_name: report_case",
                "datasets:",
                "  - name: evouna_tq",
                "    path: data/EVOUNA/TQ.json",
                "judge_models:",
                "  - name: heuristic_dummy",
                "    provider: dummy",
                "    metadata:",
                "      dummy_strategy: heuristic",
                "output:",
                f"  dir: {out}",
                "telemetry:",
                "  enabled: false",
                "  provider: null",
            ]
        ),
        encoding="utf-8",
    )
    for command in [
        ["uv", "run", "judge-eval", "prepare-data", str(config), "--sample-size", "15", "--seed", "7"],
        ["uv", "run", "judge-eval", "run", str(config), "--sample-size", "15", "--seed", "7"],
        ["uv", "run", "judge-eval", "metrics", str(out), "--bootstrap-iterations", "5"],
        ["uv", "run", "judge-eval", "report", str(out)],
    ]:
        completed = run(command, check=False, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr
    report = (out / "report.md").read_text(encoding="utf-8")
    for needle in [
        "Best overall judge",
        "Best strict-gate judge",
        "Best low-cost judge",
        "Most lenient judge",
        "Most conservative judge",
        "Most prompt-sensitive judge",
        "Worst dummy-answer robustness",
        "## Model Weaknesses",
        "## Operational Readiness",
    ]:
        assert needle in report
