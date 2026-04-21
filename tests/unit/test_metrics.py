import pandas as pd

from judge_eval.metrics import compute_metric_result, scotts_pi


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
