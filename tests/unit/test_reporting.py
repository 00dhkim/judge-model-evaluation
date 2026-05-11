import pandas as pd

from judge_eval.reporting import (
    _has_reference_order_visual_data,
    _model_color,
    _model_prompt_labels,
    _plot_reference_order_visual,
)


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


def test_model_colors_follow_provider_palette():
    assert _model_color("gpt_5_5") == "#1f1f1f"
    assert _model_color("claude_sonnet_4_6") == "#cc785c"
    assert _model_color("gemini_3_flash") == "#34A853"
    assert _model_color("qwen_3_6_plus") == "#ff7018"


def test_model_prompt_labels_hide_single_minimal_prompt():
    frame = pd.DataFrame(
        [
            {"judge_model": "gpt_5_5", "prompt_template": "minimal"},
            {"judge_model": "gemini_3_flash", "prompt_template": "minimal"},
        ]
    )

    assert _model_prompt_labels(frame, style="newline") == ["gpt_5_5", "gemini_3_flash"]
    assert _model_prompt_labels(frame, style="compact") == ["gpt_5_5", "gemini_3_flash"]


def test_model_prompt_labels_keep_prompt_when_multiple_templates_exist():
    frame = pd.DataFrame(
        [
            {"judge_model": "gpt_5_5", "prompt_template": "minimal"},
            {"judge_model": "gpt_5_5", "prompt_template": "guideline"},
        ]
    )

    assert _model_prompt_labels(frame, style="newline") == ["gpt_5_5\n(minimal)", "gpt_5_5\n(guideline)"]
    assert _model_prompt_labels(frame, style="compact") == ["gpt_5_5:minimal", "gpt_5_5:guideline"]
