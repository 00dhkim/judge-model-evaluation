import pandas as pd

from judge_eval.reporting import _has_reference_order_visual_data, _plot_reference_order_visual


def test_plot_reference_order_visual_handles_missing_metrics():
    frame = pd.DataFrame(
        [
            {
                "judge_model": "judge-a",
                "reference_order_consistency": None,
                "label_flip_rate_by_reference_order": None,
                "eligible_sample_groups": 3,
                "skipped_single_alias_groups": 2,
                "coverage": 0.6,
            }
        ]
    )

    plot = _plot_reference_order_visual(frame)

    assert isinstance(plot, str)
    assert plot


def test_has_reference_order_visual_data_false_when_metrics_missing():
    frame = pd.DataFrame(
        [
            {
                "judge_model": "judge-a",
                "reference_order_consistency": None,
                "label_flip_rate_by_reference_order": None,
                "eligible_sample_groups": 3,
                "skipped_single_alias_groups": 2,
                "coverage": 0.6,
            }
        ]
    )

    assert _has_reference_order_visual_data(frame) is False


def test_has_reference_order_visual_data_true_when_metrics_present():
    frame = pd.DataFrame(
        [
            {
                "judge_model": "judge-a",
                "reference_order_consistency": 0.8,
                "label_flip_rate_by_reference_order": 0.2,
                "eligible_sample_groups": 3,
                "skipped_single_alias_groups": 2,
                "coverage": 0.6,
            }
        ]
    )

    assert _has_reference_order_visual_data(frame) is True
