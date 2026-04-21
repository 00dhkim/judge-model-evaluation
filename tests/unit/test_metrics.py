import pandas as pd

from judge_eval.metrics import compute_metric_result, compute_reference_order_sensitivity, scotts_pi


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
