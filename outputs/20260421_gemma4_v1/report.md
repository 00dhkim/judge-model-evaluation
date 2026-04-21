# Judge Evaluation Report

## Summary

- Best overall judge: google-gemma-4-E2B-it, prompt=minimal (-0.212)
- Best strict-gate judge: google-gemma-4-E2B-it, prompt=minimal (1.000)
- Best low-cost judge: google-gemma-4-E2B-it, prompt=minimal (cost=0.000, Scott's pi=-0.212)
- Most lenient judge: google-gemma-4-E2B-it, prompt=minimal (score_delta=0.350)
- Most conservative judge: google-gemma-4-E2B-it, prompt=minimal (score_delta=0.350)
- Most prompt-sensitive judge: google-gemma-4-E2B-it (flip_rate=0.200)
- Worst dummy-answer robustness: google-gemma-4-E2B-it (robustness_accuracy=0.230)

## Ranking Table

| ranking_name | judge_model | prompt_template | rank | metric_column | metric_value |
| --- | --- | --- | --- | --- | --- |
| primary_scotts_pi | google-gemma-4-E2B-it | minimal | 1 | scotts_pi | -0.2121212121212118 |
| percent_agreement | google-gemma-4-E2B-it | minimal | 1 | percent_agreement | 0.65 |
| absolute_score_gap | google-gemma-4-E2B-it | minimal | 1 | score_delta | 0.35 |
| false_positive_rate | google-gemma-4-E2B-it | minimal | 1 | fpr | 1.0 |
| false_negative_rate | google-gemma-4-E2B-it | minimal | 1 | fnr | 0.0 |
| precision | google-gemma-4-E2B-it | minimal | 1 | precision | 0.65 |
| recall | google-gemma-4-E2B-it | minimal | 1 | recall | 1.0 |
| f1 | google-gemma-4-E2B-it | minimal | 1 | f1 | 0.787878787878788 |
| prompt_sensitivity | google-gemma-4-E2B-it |  | 1 | label_flip_rate | 0.1999999999999999 |
| dummy_answer_robustness | google-gemma-4-E2B-it |  | 1 | robustness_accuracy | 0.2299999999999999 |
| reference_order_sensitivity | google-gemma-4-E2B-it |  | 1 | label_flip_rate_by_reference_order | 0.0769230769230769 |

## Metrics Overview

| judge_model | prompt_template | judge_score | human_score | score_delta | percent_agreement | scotts_pi | precision | recall | f1 | tp | fp | tn | fn | fpr | fnr | invalid_rate | valid_rate | avg_latency_ms | p50_latency_ms | p95_latency_ms | total_estimated_cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| google-gemma-4-E2B-it | minimal | 1.0 | 0.65 | 0.35 | 0.65 | -0.2121212121212118 | 0.65 | 1.0 | 0.787878787878788 | 13 | 7 | 0 | 0 | 1.0 | 0.0 | 0.0 | 1.0 | 737.4 | 669.5 | 1144.65 | 0.0 |

## Prompt Sensitivity

| judge_model | prompt_left | prompt_right | prompt_consistency | scotts_pi_by_prompt | metric_delta_between_prompts | label_flip_rate |
| --- | --- | --- | --- | --- | --- | --- |
| google-gemma-4-E2B-it | guideline | guideline_with_examples | 0.7 | 0.5698924731182794 | 0.15 | 0.15 |
| google-gemma-4-E2B-it | guideline | minimal | 0.7 | -0.0810810810810818 | -0.15 | 0.15 |
| google-gemma-4-E2B-it | guideline_with_examples | minimal | 0.7 | -0.1764705882352937 | -0.3 | 0.3 |

## Dummy Answer Robustness

| judge_model | dummy_class | rows | valid_rows | expected_label | robustness_accuracy | rejection_rate | yes_no_caution_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| google-gemma-4-E2B-it | empty | 20 | 20 | False | 0.0 | 0.0 | 0 |
| google-gemma-4-E2B-it | gold_answer_verbatim | 20 | 20 | True | 0.95 | 0.0 | 0 |
| google-gemma-4-E2B-it | repeat_question | 20 | 20 | False | 0.05 | 0.05 | 0 |
| google-gemma-4-E2B-it | sure | 20 | 20 | False | 0.0 | 0.0 | 0 |
| google-gemma-4-E2B-it | yes | 20 | 20 | False | 0.15 | 0.15 | 0 |

## Model Weaknesses

- Highest false-positive risk: google-gemma-4-E2B-it:minimal (FPR=1.000)
- Highest false-negative risk: google-gemma-4-E2B-it:minimal (FNR=0.000)
- Weakest dummy robustness: google-gemma-4-E2B-it (robustness_accuracy=0.230)
- Highest prompt instability: google-gemma-4-E2B-it (flip_rate=0.200)

## Operational Readiness

- Recommended operational candidate: google-gemma-4-E2B-it:minimal.
- Readiness: needs further validation.
- Invalid rate: 0.000, Scott's pi: -0.212, score_delta: 0.350.

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
