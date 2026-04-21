from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class MetricResult:
    judge_score: float
    human_score: float
    score_delta: float
    percent_agreement: float
    scotts_pi: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    tn: int
    fn: int
    fpr: float
    fnr: float
    invalid_rate: float
    valid_rate: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    total_estimated_cost: float


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def scotts_pi(labels_a: Iterable[bool], labels_b: Iterable[bool]) -> float:
    a = np.array(list(labels_a), dtype=bool)
    b = np.array(list(labels_b), dtype=bool)
    if len(a) == 0:
        return 0.0
    po = float((a == b).mean())
    p_true = float((a.sum() + b.sum()) / (2 * len(a)))
    p_false = 1.0 - p_true
    pe = p_true**2 + p_false**2
    if pe == 1.0:
        return 1.0
    return float((po - pe) / (1 - pe))


def compute_metric_result(frame: pd.DataFrame) -> MetricResult:
    total = len(frame)
    valid = frame[frame["parsed_label"].notna()].copy()
    invalid_rate = safe_div(total - len(valid), total)
    valid_rate = safe_div(len(valid), total)
    if valid.empty:
        return MetricResult(*(0.0,) * 8, 0, 0, 0, 0, 0.0, 0.0, invalid_rate, valid_rate, 0.0, 0.0, 0.0, 0.0)
    pred = valid["parsed_label"].astype(bool)
    gold = valid["human_label"].astype(bool)
    latency = valid["latency_ms"].dropna() if "latency_ms" in valid.columns else pd.Series(dtype=float)
    cost = valid["estimated_cost"].dropna() if "estimated_cost" in valid.columns else pd.Series(dtype=float)
    tp = int(((pred == True) & (gold == True)).sum())
    fp = int(((pred == True) & (gold == False)).sum())
    tn = int(((pred == False) & (gold == False)).sum())
    fn = int(((pred == False) & (gold == True)).sum())
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return MetricResult(
        judge_score=float(pred.mean()),
        human_score=float(gold.mean()),
        score_delta=float(pred.mean() - gold.mean()),
        percent_agreement=float((pred == gold).mean()),
        scotts_pi=scotts_pi(pred.tolist(), gold.tolist()),
        precision=precision,
        recall=recall,
        f1=f1,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        fpr=safe_div(fp, fp + tn),
        fnr=safe_div(fn, fn + tp),
        invalid_rate=invalid_rate,
        valid_rate=valid_rate,
        avg_latency_ms=float(latency.mean()) if not latency.empty else 0.0,
        p50_latency_ms=float(latency.quantile(0.5)) if not latency.empty else 0.0,
        p95_latency_ms=float(latency.quantile(0.95)) if not latency.empty else 0.0,
        total_estimated_cost=float(cost.sum()) if not cost.empty else 0.0,
    )


def summarize_metrics(frame: pd.DataFrame, group_by: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_by):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_payload = dict(zip(group_by, keys))
        metrics = compute_metric_result(group)
        rows.append({**key_payload, **metrics.__dict__})
    return pd.DataFrame(rows)


def compute_rankings(overall: pd.DataFrame) -> pd.DataFrame:
    if overall.empty:
        return pd.DataFrame()
    ranking_specs = [
        ("primary_scotts_pi", "scotts_pi", False),
        ("percent_agreement", "percent_agreement", False),
        ("absolute_score_gap", "score_delta", True),
        ("false_positive_rate", "fpr", True),
        ("false_negative_rate", "fnr", True),
        ("precision", "precision", False),
        ("recall", "recall", False),
        ("f1", "f1", False),
    ]
    rows: list[dict[str, object]] = []
    for metric_name, column, ascending in ranking_specs:
        ranked = overall.copy()
        ranked["metric_value"] = ranked[column].abs() if metric_name == "absolute_score_gap" else ranked[column]
        ranked = ranked.sort_values(["metric_value", "judge_model", "prompt_template"], ascending=[ascending, True, True]).reset_index(drop=True)
        for index, item in ranked.iterrows():
            rows.append(
                {
                    "ranking_name": metric_name,
                    "judge_model": item["judge_model"],
                    "prompt_template": item["prompt_template"],
                    "rank": index + 1,
                    "metric_column": column,
                    "metric_value": item["metric_value"],
                }
            )
    return pd.DataFrame(rows)


def bootstrap_confidence_intervals(frame: pd.DataFrame, iterations: int = 200, seed: int = 0) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if frame.empty:
        return pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    for judge_model, group in frame.groupby("judge_model"):
        valid = group[group["parsed_label"].notna()]
        if valid.empty:
            continue
        values: list[float] = []
        data = valid[["parsed_label", "human_label"]].to_numpy()
        for _ in range(iterations):
            sample_idx = rng.integers(0, len(data), len(data))
            sampled = data[sample_idx]
            boot = pd.DataFrame({"parsed_label": sampled[:, 0], "human_label": sampled[:, 1]})
            values.append(compute_metric_result(boot).scotts_pi)
        rows.append(
            {
                "judge_model": judge_model,
                "metric": "scotts_pi",
                "ci_low": float(np.percentile(values, 2.5)),
                "ci_high": float(np.percentile(values, 97.5)),
                "iterations": iterations,
            }
        )
    return pd.DataFrame(rows)


def compute_variant_analysis(frame: pd.DataFrame, variant_type: str) -> pd.DataFrame:
    variant_rows = frame[frame["variant_type"] == variant_type].copy()
    if variant_rows.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for judge_model, group in variant_rows.groupby("judge_model"):
        valid = group[group["parsed_label"].notna()]
        consistency = 0.0
        if "variant_group" in group.columns:
            grouped = valid.groupby("variant_group")["parsed_label"].nunique()
            consistency = float((grouped <= 1).mean()) if not grouped.empty else 0.0
        rows.append(
            {
                "judge_model": judge_model,
                "variant_type": variant_type,
                "rows": len(group),
                "valid_rows": len(valid),
                "mean_label": float(valid["parsed_label"].astype(bool).mean()) if not valid.empty else 0.0,
                "consistency": consistency,
            }
        )
    return pd.DataFrame(rows)


def compute_prompt_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    prompt_rows = frame[frame["variant_type"] == "prompt_sensitivity"].copy()
    if prompt_rows.empty:
        return pd.DataFrame(
            columns=[
                "judge_model",
                "prompt_left",
                "prompt_right",
                "prompt_consistency",
                "scotts_pi_by_prompt",
                "metric_delta_between_prompts",
                "label_flip_rate",
            ]
        )
    rows: list[dict[str, object]] = []
    for judge_model, group in prompt_rows.groupby("judge_model"):
        valid = group[group["parsed_label"].notna()].copy()
        pivot = valid.pivot_table(
            index="sample_id",
            columns="prompt_template",
            values="parsed_label",
            aggfunc="first",
        )
        if pivot.empty:
            continue
        consistency = float((pivot.nunique(axis=1) <= 1).mean())
        for left in pivot.columns:
            for right in pivot.columns:
                if left >= right:
                    continue
                pair = pivot[[left, right]].dropna()
                if pair.empty:
                    continue
                pi_value = scotts_pi(pair[left].astype(bool).tolist(), pair[right].astype(bool).tolist())
                delta = float((pair[left].astype(bool).mean()) - (pair[right].astype(bool).mean()))
                rows.append(
                    {
                        "judge_model": judge_model,
                        "prompt_left": left,
                        "prompt_right": right,
                        "prompt_consistency": consistency,
                        "scotts_pi_by_prompt": pi_value,
                        "metric_delta_between_prompts": delta,
                        "label_flip_rate": float((pair[left] != pair[right]).mean()),
                    }
                )
    return pd.DataFrame(rows)


def compute_reference_order_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    variants = frame[frame["variant_type"] == "reference_order"].copy()
    if variants.empty:
        return pd.DataFrame(
            columns=[
                "judge_model",
                "reference_order_consistency",
                "label_flip_rate_by_reference_order",
                "rows",
                "valid_rows",
            ]
        )
    for judge_model, group in variants.groupby("judge_model"):
        valid = group[group["parsed_label"].notna()]
        grouped = valid.groupby("variant_group")["parsed_label"]
        consistency = float((grouped.nunique() <= 1).mean()) if len(valid) else 0.0
        flip_rate = float((grouped.nunique() > 1).mean()) if len(valid) else 0.0
        rows.append(
            {
                "judge_model": judge_model,
                "reference_order_consistency": consistency,
                "label_flip_rate_by_reference_order": flip_rate,
                "rows": len(group),
                "valid_rows": len(valid),
            }
        )
    return pd.DataFrame(rows)


def compute_dummy_answer_robustness(frame: pd.DataFrame) -> pd.DataFrame:
    variants = frame[frame["variant_type"] == "dummy_answer"].copy()
    if variants.empty:
        return pd.DataFrame(
            columns=[
                "judge_model",
                "dummy_class",
                "rows",
                "valid_rows",
                "expected_label",
                "robustness_accuracy",
                "rejection_rate",
                "yes_no_caution_count",
            ]
        )
    rows = []
    for (judge_model, variant_id), group in variants.groupby(["judge_model", "variant_id"]):
        valid = group[group["parsed_label"].notna()].copy()
        if valid.empty:
            rejection_rate = 0.0
            accuracy = 0.0
        else:
            if variant_id == "gold_answer_verbatim":
                rejection_rate = 0.0
            else:
                rejection_rate = float((~valid["parsed_label"].astype(bool)).mean())
            accuracy = float((valid["parsed_label"].astype(bool) == valid["human_label"].astype(bool)).mean())
        rows.append(
            {
                "judge_model": judge_model,
                "dummy_class": variant_id,
                "rows": len(group),
                "valid_rows": len(valid),
                "expected_label": bool(group["human_label"].iloc[0]),
                "robustness_accuracy": accuracy,
                "rejection_rate": rejection_rate,
                "yes_no_caution_count": int(
                    sum(1 for item in group["variant_metadata"] if isinstance(item, dict) and item.get("yes_no_caution"))
                ),
            }
        )
    return pd.DataFrame(rows)


def write_metrics_bundle(parsed: pd.DataFrame, output_dir: Path, bootstrap_iterations: int) -> None:
    base = parsed[parsed["variant_type"] == "base"].copy()
    overall = summarize_metrics(base, ["judge_model", "prompt_template"])
    by_dataset = summarize_metrics(base, ["judge_model", "dataset"])
    by_answer_source = summarize_metrics(base, ["judge_model", "answer_source"])
    by_human_label = summarize_metrics(base, ["judge_model", "human_label"])
    by_length = summarize_metrics(base, ["judge_model", "answer_length_bucket"])
    by_alias_count = summarize_metrics(base, ["judge_model", "golden_answer_alias_count"])
    by_model_family = summarize_metrics(base, ["judge_model", "model_family"])
    overall.to_csv(output_dir / "metrics_overall.csv", index=False)
    by_dataset.to_csv(output_dir / "metrics_by_dataset.csv", index=False)
    by_answer_source.to_csv(output_dir / "metrics_by_answer_source.csv", index=False)
    by_human_label.to_csv(output_dir / "metrics_by_human_label.csv", index=False)
    by_length.to_csv(output_dir / "metrics_by_answer_length_bucket.csv", index=False)
    by_alias_count.to_csv(output_dir / "metrics_by_alias_count.csv", index=False)
    by_model_family.to_csv(output_dir / "metrics_by_model_family.csv", index=False)
    overall[["judge_model", "prompt_template", "tp", "fp", "tn", "fn"]].to_json(
        output_dir / "confusion_matrices.json",
        orient="records",
        indent=2,
    )
    overall[["judge_model", "prompt_template", "scotts_pi"]].to_csv(output_dir / "scotts_pi.csv", index=False)
    prompt_sensitivity = compute_prompt_sensitivity(parsed)
    reference_order = compute_reference_order_sensitivity(parsed)
    dummy_robustness = compute_dummy_answer_robustness(parsed)
    prompt_sensitivity.to_csv(output_dir / "prompt_sensitivity.csv", index=False)
    reference_order.to_csv(output_dir / "reference_order_sensitivity.csv", index=False)
    dummy_robustness.to_csv(output_dir / "dummy_answer_robustness.csv", index=False)
    leniency = overall[["judge_model", "prompt_template", "judge_score", "human_score", "score_delta", "fpr", "fnr"]]
    leniency.to_csv(output_dir / "leniency_bias.csv", index=False)
    bootstrap_confidence_intervals(base, iterations=bootstrap_iterations).to_csv(
        output_dir / "bootstrap_confidence_intervals.csv",
        index=False,
    )
    rankings = compute_rankings(overall)
    additional_rows: list[dict[str, object]] = []
    if not prompt_sensitivity.empty:
        prompt_rank = (
            prompt_sensitivity.groupby("judge_model", as_index=False)["label_flip_rate"]
            .mean()
            .sort_values(["label_flip_rate", "judge_model"], ascending=[True, True])
            .reset_index(drop=True)
        )
        for index, item in prompt_rank.iterrows():
            additional_rows.append(
                {
                    "ranking_name": "prompt_sensitivity",
                    "judge_model": item["judge_model"],
                    "prompt_template": "",
                    "rank": index + 1,
                    "metric_column": "label_flip_rate",
                    "metric_value": item["label_flip_rate"],
                }
            )
    if not dummy_robustness.empty:
        dummy_rank = (
            dummy_robustness.groupby("judge_model", as_index=False)["robustness_accuracy"]
            .mean()
            .sort_values(["robustness_accuracy", "judge_model"], ascending=[False, True])
            .reset_index(drop=True)
        )
        for index, item in dummy_rank.iterrows():
            additional_rows.append(
                {
                    "ranking_name": "dummy_answer_robustness",
                    "judge_model": item["judge_model"],
                    "prompt_template": "",
                    "rank": index + 1,
                    "metric_column": "robustness_accuracy",
                    "metric_value": item["robustness_accuracy"],
                }
            )
    if not reference_order.empty:
        ref_rank = reference_order.sort_values(
            ["label_flip_rate_by_reference_order", "judge_model"],
            ascending=[True, True],
        ).reset_index(drop=True)
        for index, item in ref_rank.iterrows():
            additional_rows.append(
                {
                    "ranking_name": "reference_order_sensitivity",
                    "judge_model": item["judge_model"],
                    "prompt_template": "",
                    "rank": index + 1,
                    "metric_column": "label_flip_rate_by_reference_order",
                    "metric_value": item["label_flip_rate_by_reference_order"],
                }
            )
    if additional_rows:
        rankings = pd.concat([rankings, pd.DataFrame(additional_rows)], ignore_index=True)
    rankings.to_csv(output_dir / "model_rankings.csv", index=False)
