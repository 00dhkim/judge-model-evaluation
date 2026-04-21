from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pandas.errors import EmptyDataError

from judge_eval.utils import ensure_dir


def _frame_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.fillna("").astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def generate_report(output_dir: Path) -> Path:
    metrics = _safe_read_csv(output_dir / "metrics_overall.csv")
    rankings = _safe_read_csv(output_dir / "model_rankings.csv")
    prompt_sensitivity = _safe_read_csv(output_dir / "prompt_sensitivity.csv")
    dummy_robustness = _safe_read_csv(output_dir / "dummy_answer_robustness.csv")
    answer_source_metrics = _safe_read_csv(output_dir / "metrics_by_answer_source.csv")
    leniency = _safe_read_csv(output_dir / "leniency_bias.csv")
    plots_dir = ensure_dir(output_dir / "plots")
    if not metrics.empty:
        figure = plt.figure(figsize=(10, 5))
        axes = figure.add_subplot(111)
        axes.bar(metrics["judge_model"] + ":" + metrics["prompt_template"], metrics["scotts_pi"])
        axes.set_title("Scott's Pi by Judge Model / Prompt")
        axes.set_ylabel("Scott's Pi")
        axes.tick_params(axis="x", rotation=45)
        figure.tight_layout()
        figure.savefig(plots_dir / "scotts_pi.png")
        plt.close(figure)
        figure = plt.figure(figsize=(10, 5))
        axes = figure.add_subplot(111)
        axes.bar(metrics["judge_model"] + ":" + metrics["prompt_template"], metrics["percent_agreement"])
        axes.set_title("Percent Agreement by Judge Model / Prompt")
        axes.tick_params(axis="x", rotation=45)
        figure.tight_layout()
        figure.savefig(plots_dir / "percent_agreement.png")
        plt.close(figure)
        figure = plt.figure(figsize=(10, 5))
        axes = figure.add_subplot(111)
        axes.bar(metrics["judge_model"] + ":" + metrics["prompt_template"], metrics["score_delta"])
        axes.set_title("Judge score gap by Judge Model / Prompt")
        axes.tick_params(axis="x", rotation=45)
        figure.tight_layout()
        figure.savefig(plots_dir / "score_gap.png")
        plt.close(figure)
        figure = plt.figure(figsize=(8, 5))
        axes = figure.add_subplot(111)
        axes.scatter(metrics["human_score"], metrics["judge_score"])
        axes.set_xlabel("Human score")
        axes.set_ylabel("Judge score")
        axes.set_title("Judge score vs Human score")
        figure.tight_layout()
        figure.savefig(plots_dir / "judge_vs_human_score.png")
        plt.close(figure)
        figure = plt.figure(figsize=(10, 5))
        axes = figure.add_subplot(111)
        width = 0.25
        x = range(len(metrics))
        axes.bar([item - width for item in x], metrics["precision"], width=width, label="Precision")
        axes.bar(list(x), metrics["recall"], width=width, label="Recall")
        axes.bar([item + width for item in x], metrics["f1"], width=width, label="F1")
        axes.set_xticks(list(x))
        axes.set_xticklabels((metrics["judge_model"] + ":" + metrics["prompt_template"]).tolist(), rotation=45)
        axes.set_title("Precision / Recall / F1")
        axes.legend()
        figure.tight_layout()
        figure.savefig(plots_dir / "precision_recall_f1.png")
        plt.close(figure)
        figure = plt.figure(figsize=(10, 5))
        axes = figure.add_subplot(111)
        width = 0.35
        x = range(len(metrics))
        axes.bar([item - width / 2 for item in x], metrics["fpr"], width=width, label="FPR")
        axes.bar([item + width / 2 for item in x], metrics["fnr"], width=width, label="FNR")
        axes.set_xticks(list(x))
        axes.set_xticklabels((metrics["judge_model"] + ":" + metrics["prompt_template"]).tolist(), rotation=45)
        axes.set_title("False Positive / False Negative Rates")
        axes.legend()
        figure.tight_layout()
        figure.savefig(plots_dir / "fpr_fnr.png")
        plt.close(figure)
        figure = plt.figure(figsize=(8, 5))
        axes = figure.add_subplot(111)
        axes.scatter(
            metrics["avg_latency_ms"] if "avg_latency_ms" in metrics.columns else [0] * len(metrics),
            metrics["total_estimated_cost"] if "total_estimated_cost" in metrics.columns else [0] * len(metrics),
        )
        axes.set_xlabel("Latency (ms)")
        axes.set_ylabel("Estimated cost")
        axes.set_title("Cost / Latency Trade-off")
        figure.tight_layout()
        figure.savefig(plots_dir / "cost_latency_tradeoff.png")
        plt.close(figure)
    if not prompt_sensitivity.empty:
        pivot = prompt_sensitivity.pivot_table(
            index="judge_model",
            columns="prompt_left",
            values="label_flip_rate",
            aggfunc="mean",
            fill_value=0.0,
        )
        figure = plt.figure(figsize=(8, 5))
        axes = figure.add_subplot(111)
        im = axes.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
        axes.set_xticks(range(len(pivot.columns)))
        axes.set_xticklabels(pivot.columns.tolist(), rotation=45)
        axes.set_yticks(range(len(pivot.index)))
        axes.set_yticklabels(pivot.index.tolist())
        axes.set_title("Prompt Sensitivity Heatmap")
        figure.colorbar(im, ax=axes)
        figure.tight_layout()
        figure.savefig(plots_dir / "prompt_sensitivity_heatmap.png")
        plt.close(figure)
    if not answer_source_metrics.empty:
        pivot = answer_source_metrics.pivot_table(
            index="judge_model",
            columns="answer_source",
            values="scotts_pi",
            aggfunc="mean",
            fill_value=0.0,
        )
        figure = plt.figure(figsize=(8, 5))
        axes = figure.add_subplot(111)
        im = axes.imshow(pivot.to_numpy(), aspect="auto", cmap="magma")
        axes.set_xticks(range(len(pivot.columns)))
        axes.set_xticklabels(pivot.columns.tolist(), rotation=45)
        axes.set_yticks(range(len(pivot.index)))
        axes.set_yticklabels(pivot.index.tolist())
        axes.set_title("Answer Source Scott's Pi Heatmap")
        figure.colorbar(im, ax=axes)
        figure.tight_layout()
        figure.savefig(plots_dir / "answer_source_scotts_pi_heatmap.png")
        plt.close(figure)
    if not dummy_robustness.empty:
        figure = plt.figure(figsize=(10, 5))
        axes = figure.add_subplot(111)
        labels = dummy_robustness["judge_model"] + ":" + dummy_robustness["dummy_class"]
        axes.bar(labels, dummy_robustness["robustness_accuracy"])
        axes.set_title("Dummy Answer Robustness")
        axes.tick_params(axis="x", rotation=45)
        figure.tight_layout()
        figure.savefig(plots_dir / "dummy_answer_robustness.png")
        plt.close(figure)
    callouts = _build_callouts(metrics, rankings, prompt_sensitivity, dummy_robustness, leniency)
    report = "\n".join(
        [
            "# Judge Evaluation Report",
            "",
            "## Summary",
            "",
            callouts,
            "",
            "## Ranking Table",
            "",
            _frame_to_markdown(rankings) if not rankings.empty else "No ranking rows.",
            "",
            "## Metrics Overview",
            "",
            _frame_to_markdown(metrics) if not metrics.empty else "No metric rows.",
            "",
            "## Prompt Sensitivity",
            "",
            _frame_to_markdown(prompt_sensitivity) if not prompt_sensitivity.empty else "No prompt sensitivity rows.",
            "",
            "## Dummy Answer Robustness",
            "",
            _frame_to_markdown(dummy_robustness) if not dummy_robustness.empty else "No dummy robustness rows.",
            "",
            "## Model Weaknesses",
            "",
            _model_weaknesses(metrics, dummy_robustness, prompt_sensitivity),
            "",
            "## Operational Readiness",
            "",
            _operational_readiness(metrics, leniency),
            "",
            "## Plots",
            "",
            "- `plots/scotts_pi.png`",
            "- `plots/percent_agreement.png`",
            "- `plots/score_gap.png`",
            "- `plots/judge_vs_human_score.png`",
            "- `plots/precision_recall_f1.png`",
            "- `plots/fpr_fnr.png`",
            "- `plots/prompt_sensitivity_heatmap.png`",
            "- `plots/answer_source_scotts_pi_heatmap.png`",
            "- `plots/dummy_answer_robustness.png`",
            "- `plots/cost_latency_tradeoff.png`",
            "",
            "## Operational Notes",
            "",
            "- Telemetry manifest is stored in `telemetry_manifest.json`.",
            "- Invalid outputs are reported explicitly and never coerced into booleans.",
        ]
    )
    report_path = output_dir / "report.md"
    report_path.write_text(report + "\n", encoding="utf-8")
    return report_path


def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def _ranking_winner(rankings: pd.DataFrame, ranking_name: str) -> str:
    rows = rankings[rankings["ranking_name"] == ranking_name]
    if rows.empty:
        return "n/a"
    winner = rows.sort_values(["rank", "judge_model"]).iloc[0]
    prompt = f", prompt={winner['prompt_template']}" if winner.get("prompt_template") else ""
    return f"{winner['judge_model']}{prompt} ({winner['metric_value']:.3f})"


def _build_callouts(
    metrics: pd.DataFrame,
    rankings: pd.DataFrame,
    prompt_sensitivity: pd.DataFrame,
    dummy_robustness: pd.DataFrame,
    leniency: pd.DataFrame,
) -> str:
    lines = [
        f"- Best overall judge: {_ranking_winner(rankings, 'primary_scotts_pi')}",
        f"- Best strict-gate judge: {_ranking_winner(rankings, 'false_positive_rate')}",
        f"- Best low-cost judge: {_low_cost_callout(metrics)}",
        f"- Most lenient judge: {_leniency_callout(leniency, most_lenient=True)}",
        f"- Most conservative judge: {_leniency_callout(leniency, most_lenient=False)}",
        f"- Most prompt-sensitive judge: {_prompt_callout(prompt_sensitivity)}",
        f"- Worst dummy-answer robustness: {_dummy_callout(dummy_robustness)}",
    ]
    return "\n".join(lines)


def _low_cost_callout(metrics: pd.DataFrame) -> str:
    if metrics.empty:
        return "n/a"
    frame = metrics.copy()
    frame["total_estimated_cost"] = frame["total_estimated_cost"].fillna(0.0)
    frame = frame.sort_values(["total_estimated_cost", "scotts_pi"], ascending=[True, False])
    row = frame.iloc[0]
    return f"{row['judge_model']}, prompt={row['prompt_template']} (cost={row['total_estimated_cost']:.3f}, Scott's pi={row['scotts_pi']:.3f})"


def _leniency_callout(leniency: pd.DataFrame, most_lenient: bool) -> str:
    if leniency.empty:
        return "n/a"
    row = leniency.sort_values("score_delta", ascending=not most_lenient).iloc[0]
    return f"{row['judge_model']}, prompt={row['prompt_template']} (score_delta={row['score_delta']:.3f})"


def _prompt_callout(prompt_sensitivity: pd.DataFrame) -> str:
    if prompt_sensitivity.empty:
        return "n/a"
    grouped = prompt_sensitivity.groupby("judge_model", as_index=False)["label_flip_rate"].mean()
    row = grouped.sort_values("label_flip_rate", ascending=False).iloc[0]
    return f"{row['judge_model']} (flip_rate={row['label_flip_rate']:.3f})"


def _dummy_callout(dummy_robustness: pd.DataFrame) -> str:
    if dummy_robustness.empty:
        return "n/a"
    grouped = dummy_robustness.groupby("judge_model", as_index=False)["robustness_accuracy"].mean()
    row = grouped.sort_values("robustness_accuracy", ascending=True).iloc[0]
    return f"{row['judge_model']} (robustness_accuracy={row['robustness_accuracy']:.3f})"


def _model_weaknesses(metrics: pd.DataFrame, dummy_robustness: pd.DataFrame, prompt_sensitivity: pd.DataFrame) -> str:
    if metrics.empty:
        return "No model weaknesses available."
    lines = []
    worst_fpr = metrics.sort_values("fpr", ascending=False).iloc[0]
    lines.append(f"- Highest false-positive risk: {worst_fpr['judge_model']}:{worst_fpr['prompt_template']} (FPR={worst_fpr['fpr']:.3f})")
    worst_fnr = metrics.sort_values("fnr", ascending=False).iloc[0]
    lines.append(f"- Highest false-negative risk: {worst_fnr['judge_model']}:{worst_fnr['prompt_template']} (FNR={worst_fnr['fnr']:.3f})")
    if not dummy_robustness.empty:
        lines.append(f"- Weakest dummy robustness: {_dummy_callout(dummy_robustness)}")
    if not prompt_sensitivity.empty:
        lines.append(f"- Highest prompt instability: {_prompt_callout(prompt_sensitivity)}")
    return "\n".join(lines)


def _operational_readiness(metrics: pd.DataFrame, leniency: pd.DataFrame) -> str:
    if metrics.empty:
        return "No operational readiness data available."
    row = metrics.sort_values(["invalid_rate", "scotts_pi"], ascending=[True, False]).iloc[0]
    readiness = "ready for guarded use" if row["invalid_rate"] < 0.05 and row["scotts_pi"] >= 0.5 else "needs further validation"
    return (
        f"- Recommended operational candidate: {row['judge_model']}:{row['prompt_template']}.\n"
        f"- Readiness: {readiness}.\n"
        f"- Invalid rate: {row['invalid_rate']:.3f}, Scott's pi: {row['scotts_pi']:.3f}, score_delta: {row['score_delta']:.3f}."
    )
