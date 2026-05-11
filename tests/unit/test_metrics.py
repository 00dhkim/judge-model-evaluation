import pandas as pd

from judge_eval.metrics import (
    compute_metric_result,
    compute_reference_order_sensitivity,
    format_evaluation_coverage_warning,
    scotts_pi,
    write_metrics_bundle,
)


def test_scotts_pi_perfect_agreement():
    assert scotts_pi([True, False], [True, False]) == 1.0


def test_compute_metric_result_counts():
    frame = pd.DataFrame(
        [
            {"parsed_label": True, "human_label": True},
            {"parsed_label": True, "human_label": False},
            {"parsed_label": False, "human_label": False},
            {"parsed_label": None, "human_label": True},
        ]
    )
    result = compute_metric_result(frame)
    assert result.tp == 1
    assert result.fp == 1
    assert result.tn == 1
    assert result.fn == 0
    assert result.invalid_rate == 0.25


def test_compute_reference_order_sensitivity_reports_coverage():
    frame = pd.DataFrame(
        [
            {
                "judge_model": "judge-a",
                "variant_type": "base",
                "variant_group": "s1",
                "golden_answer_alias_count": 1,
                "parsed_label": True,
            },
            {
                "judge_model": "judge-a",
                "variant_type": "base",
                "variant_group": "s2",
                "golden_answer_alias_count": 3,
                "parsed_label": True,
            },
            {
                "judge_model": "judge-a",
                "variant_type": "reference_order",
                "variant_group": "s2",
                "golden_answer_alias_count": 3,
                "parsed_label": True,
            },
            {
                "judge_model": "judge-a",
                "variant_type": "reference_order",
                "variant_group": "s2",
                "golden_answer_alias_count": 3,
                "parsed_label": False,
            },
        ]
    )
    result = compute_reference_order_sensitivity(frame)
    row = result.iloc[0]
    assert row["eligible_sample_groups"] == 1
    assert row["skipped_single_alias_groups"] == 1
    assert row["coverage"] == 0.5
    assert row["label_flip_rate_by_reference_order"] == 1.0


def test_format_evaluation_coverage_warning_returns_none_for_matching_coverage():
    frame = pd.DataFrame(
        [
            {"judge_model": "judge-a", "prompt_template": "minimal", "parsed_label": True, "human_label": True},
            {"judge_model": "judge-a", "prompt_template": "minimal", "parsed_label": False, "human_label": False},
            {"judge_model": "judge-b", "prompt_template": "minimal", "parsed_label": True, "human_label": True},
            {"judge_model": "judge-b", "prompt_template": "minimal", "parsed_label": True, "human_label": False},
        ]
    )

    assert format_evaluation_coverage_warning(frame) is None


def test_format_evaluation_coverage_warning_reports_mismatched_human_distribution():
    frame = pd.DataFrame(
        [
            {"judge_model": "judge-a", "prompt_template": "minimal", "parsed_label": True, "human_label": True},
            {"judge_model": "judge-a", "prompt_template": "minimal", "parsed_label": False, "human_label": False},
            {"judge_model": "judge-b", "prompt_template": "minimal", "parsed_label": True, "human_label": True},
            {"judge_model": "judge-b", "prompt_template": "minimal", "parsed_label": None, "human_label": True},
        ]
    )

    warning = format_evaluation_coverage_warning(frame)

    assert warning is not None
    assert warning.startswith("Warning: evaluation coverage mismatch for prompt_template(s): minimal;")
    assert "judge-a" in warning
    assert "judge-b" in warning
    assert "human_positive_valid" not in warning
    assert "human_negative_valid" not in warning
    assert "human_positive_total" not in warning
    assert "human_negative_total" not in warning
    assert "\033[33m    judge-b" in warning


def test_write_metrics_bundle_prints_coverage_warning(tmp_path, capsys):
    frame = pd.DataFrame(
        [
            {
                "judge_model": "judge-a",
                "prompt_template": "minimal",
                "variant_type": "base",
                "variant_group": "sample-1",
                "parsed_label": True,
                "human_label": True,
                "latency_ms": 1,
                "estimated_cost": 0,
                "dataset": "TQ",
                "answer_source": "fid",
                "answer_length_bucket": "short",
                "golden_answer_alias_count": 1,
                "model_family": "dummy",
            },
            {
                "judge_model": "judge-b",
                "prompt_template": "minimal",
                "variant_type": "base",
                "variant_group": "sample-1",
                "parsed_label": True,
                "human_label": False,
                "latency_ms": 1,
                "estimated_cost": 0,
                "dataset": "TQ",
                "answer_source": "fid",
                "answer_length_bucket": "short",
                "golden_answer_alias_count": 1,
                "model_family": "dummy",
            },
        ]
    )

    write_metrics_bundle(frame, tmp_path, bootstrap_iterations=1)

    captured = capsys.readouterr()
    assert "evaluation coverage mismatch" in captured.err
    assert "judge-a" in captured.err
    assert "judge-b" in captured.err
