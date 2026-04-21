from pathlib import Path
from subprocess import run

from judge_eval.config import config_hash, load_config, resolve_output_dir


def test_report_contains_required_sections(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "experiment_name: report_case",
                "datasets:",
                "  - name: evouna_tq",
                "    path: data/EVOUNA/TQ.json",
                "    sampling:",
                "      sample_size: 15",
                "      seed: 7",
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
    loaded_config, _ = load_config(config)
    out = resolve_output_dir(loaded_config, config_hash(config))
    for command in [
        ["uv", "run", "judge-eval", "prepare-data", str(config)],
        ["uv", "run", "judge-eval", "run", str(config)],
        ["uv", "run", "judge-eval", "metrics", str(out), "--bootstrap-iterations", "5"],
        ["uv", "run", "judge-eval", "report", str(out)],
    ]:
        completed = run(command, check=False, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr
    report = (out / "report.html").read_text(encoding="utf-8")
    for needle in [
        "최우수 judge (Scott's π)",
        "엄격 게이트 최우수 (FPR 최소)",
        "최저 비용 judge",
        "가장 관대한 judge",
        "가장 엄격한 judge",
        "프롬프트 민감도 최고",
        "더미 강건성 최저",
        "Reference Order 적용 범위",
        "운영 투입 판단",
    ]:
        assert needle in report
