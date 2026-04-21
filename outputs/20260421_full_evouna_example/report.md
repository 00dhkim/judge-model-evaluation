# Judge Evaluation Report

## Summary

- Best overall judge: perfect_dummy, prompt=minimal (0.683)
- Best strict-gate judge: invalid_dummy, prompt=minimal (0.000)
- Best low-cost judge: perfect_dummy, prompt=minimal (cost=0.000, Scott's pi=0.683)
- Most lenient judge: invalid_dummy, prompt=minimal (score_delta=0.000)
- Most conservative judge: perfect_dummy, prompt=minimal (score_delta=-0.100)
- Most prompt-sensitive judge: perfect_dummy (flip_rate=0.000)
- Worst dummy-answer robustness: invalid_dummy (robustness_accuracy=0.000)

## Ranking Table

| ranking_name | judge_model | prompt_template | rank | metric_column | metric_value |
| --- | --- | --- | --- | --- | --- |
| primary_scotts_pi | perfect_dummy | minimal | 1 | scotts_pi | 0.6834011759384893 |
| primary_scotts_pi | invalid_dummy | minimal | 2 | scotts_pi | 0.0 |
| percent_agreement | perfect_dummy | minimal | 1 | percent_agreement | 0.86 |
| percent_agreement | invalid_dummy | minimal | 2 | percent_agreement | 0.0 |
| absolute_score_gap | invalid_dummy | minimal | 1 | score_delta | 0.0 |
| absolute_score_gap | perfect_dummy | minimal | 2 | score_delta | 0.0999999999999999 |
| false_positive_rate | invalid_dummy | minimal | 1 | fpr | 0.0 |
| false_positive_rate | perfect_dummy | minimal | 2 | fpr | 0.0714285714285714 |
| false_negative_rate | invalid_dummy | minimal | 1 | fnr | 0.0 |
| false_negative_rate | perfect_dummy | minimal | 2 | fnr | 0.1666666666666666 |
| precision | perfect_dummy | minimal | 1 | precision | 0.967741935483871 |
| precision | invalid_dummy | minimal | 2 | precision | 0.0 |
| recall | perfect_dummy | minimal | 1 | recall | 0.8333333333333334 |
| recall | invalid_dummy | minimal | 2 | recall | 0.0 |
| f1 | perfect_dummy | minimal | 1 | f1 | 0.8955223880597015 |
| f1 | invalid_dummy | minimal | 2 | f1 | 0.0 |
| prompt_sensitivity | perfect_dummy |  | 1 | label_flip_rate | 0.0 |
| dummy_answer_robustness | perfect_dummy |  | 1 | robustness_accuracy | 0.796 |
| dummy_answer_robustness | invalid_dummy |  | 2 | robustness_accuracy | 0.0 |
| reference_order_sensitivity | invalid_dummy |  | 1 | label_flip_rate_by_reference_order | 0.0 |
| reference_order_sensitivity | perfect_dummy |  | 2 | label_flip_rate_by_reference_order | 0.0 |

## Metrics Overview

| judge_model | prompt_template | judge_score | human_score | score_delta | percent_agreement | scotts_pi | precision | recall | f1 | tp | fp | tn | fn | fpr | fnr | invalid_rate | valid_rate | avg_latency_ms | p50_latency_ms | p95_latency_ms | total_estimated_cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| invalid_dummy | minimal | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 | 0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| perfect_dummy | minimal | 0.62 | 0.72 | -0.0999999999999999 | 0.86 | 0.6834011759384893 | 0.967741935483871 | 0.8333333333333334 | 0.8955223880597015 | 30 | 1 | 13 | 6 | 0.0714285714285714 | 0.1666666666666666 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## Prompt Sensitivity

| judge_model | prompt_left | prompt_right | prompt_consistency | scotts_pi_by_prompt | metric_delta_between_prompts | label_flip_rate |
| --- | --- | --- | --- | --- | --- | --- |
| perfect_dummy | guideline | guideline_with_examples | 1.0 | 1.0 | 0.0 | 0.0 |
| perfect_dummy | guideline | minimal | 1.0 | 1.0 | 0.0 | 0.0 |
| perfect_dummy | guideline_with_examples | minimal | 1.0 | 1.0 | 0.0 | 0.0 |

## Dummy Answer Robustness

| judge_model | dummy_class | rows | valid_rows | expected_label | robustness_accuracy | rejection_rate | yes_no_caution_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| invalid_dummy | empty | 50 | 0 | False | 0.0 | 0.0 | 0 |
| invalid_dummy | gold_answer_verbatim | 50 | 0 | True | 0.0 | 0.0 | 0 |
| invalid_dummy | repeat_question | 50 | 0 | False | 0.0 | 0.0 | 0 |
| invalid_dummy | sure | 50 | 0 | False | 0.0 | 0.0 | 0 |
| invalid_dummy | yes | 50 | 0 | False | 0.0 | 0.0 | 1 |
| perfect_dummy | empty | 50 | 50 | False | 0.0 | 0.0 | 0 |
| perfect_dummy | gold_answer_verbatim | 50 | 50 | True | 1.0 | 0.0 | 0 |
| perfect_dummy | repeat_question | 50 | 50 | False | 1.0 | 1.0 | 0 |
| perfect_dummy | sure | 50 | 50 | False | 1.0 | 1.0 | 0 |
| perfect_dummy | yes | 50 | 50 | False | 0.98 | 1.0 | 1 |

## Model Weaknesses

- Highest false-positive risk: perfect_dummy:minimal (FPR=0.071)
- Highest false-negative risk: perfect_dummy:minimal (FNR=0.167)
- Weakest dummy robustness: invalid_dummy (robustness_accuracy=0.000)
- Highest prompt instability: perfect_dummy (flip_rate=0.000)

## Operational Readiness

- Recommended operational candidate: perfect_dummy:minimal.
- Readiness: ready for guarded use.
- Invalid rate: 0.000, Scott's pi: 0.683, score_delta: -0.100.

## Plots

- `plots/scotts_pi.png`
- `plots/percent_agreement.png`
- `plots/score_gap.png`
- `plots/judge_vs_human_score.png`
- `plots/precision_recall_f1.png`
- `plots/fpr_fnr.png`
- `plots/prompt_sensitivity_heatmap.png`
- `plots/answer_source_scotts_pi_heatmap.png`
- `plots/dummy_answer_robustness.png`
- `plots/cost_latency_tradeoff.png`

## Operational Notes

- Telemetry manifest is stored in `telemetry_manifest.json`.
- Invalid outputs are reported explicitly and never coerced into booleans.
