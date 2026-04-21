from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from judge_eval.utils import ensure_dir

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Paper reference data (Bavaresco et al., 2024 — "Judging the Judges",
# arXiv:2406.12624). Values are approximate TriviaQA results extracted from
# the paper. Dataset differs from EVOUNA, so treat as directional reference only.
# ---------------------------------------------------------------------------
PAPER_REFERENCE = {
    "model": [
        "GPT-4-Turbo",
        "GPT-4",
        "Claude-3-Opus",
        "Llama-3-70B",
        "GPT-3.5-Turbo",
        "Mixtral-8x7B",
        "Llama-2-70B",
    ],
    "scotts_pi": [0.82, 0.79, 0.77, 0.67, 0.60, 0.48, 0.37],
    "percent_agreement": [0.92, 0.91, 0.90, 0.86, 0.84, 0.80, 0.75],
    "fpr": [0.07, 0.09, 0.11, 0.14, 0.18, 0.22, 0.29],
    "fnr": [0.06, 0.08, 0.08, 0.12, 0.15, 0.18, 0.25],
    "precision": [0.93, 0.91, 0.90, 0.87, 0.84, 0.80, 0.74],
    "recall": [0.94, 0.92, 0.92, 0.88, 0.85, 0.82, 0.75],
    "f1": [0.935, 0.915, 0.910, 0.875, 0.845, 0.810, 0.745],
}

METRIC_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "scotts_pi": {
        "title": "Scott's π (Chance-Corrected Agreement)",
        "desc": (
            "인간 평가자와 judge 모델 간의 레이블 일치율에서 우연히 맞을 확률을 보정한 지표입니다. "
            "단순 percent agreement보다 클래스 불균형에 강인합니다."
        ),
        "high": "높을수록 좋음 — 1.0에 가까울수록 인간 판단과 완벽히 일치",
        "low": "낮거나 음수 — 우연보다 나쁘거나 체계적으로 반대 판단을 하는 것",
        "threshold": "≥ 0.6 이면 실용적 사용 가능, ≥ 0.8 이면 우수",
    },
    "percent_agreement": {
        "title": "Percent Agreement (단순 일치율)",
        "desc": "Judge 모델 레이블과 인간 레이블이 일치하는 샘플의 비율입니다.",
        "high": "높을수록 좋음 — 1.0이면 모든 샘플에서 인간과 동일 판단",
        "low": "낮을수록 — 오분류가 많음. 클래스 불균형 시 높아도 Scott's π가 낮을 수 있음",
        "threshold": "≥ 0.85 이면 양호",
    },
    "precision_recall_f1": {
        "title": "Precision / Recall / F1",
        "desc": (
            "Precision: judge가 '정답'이라 한 것 중 실제 정답 비율. "
            "Recall: 실제 정답 중 judge가 '정답'으로 잡은 비율. "
            "F1: 둘의 조화평균."
        ),
        "high": "모두 높을수록 좋음 — 정밀도와 재현율의 균형이 이상적",
        "low": "Precision이 낮으면 오탐이 많고, Recall이 낮으면 정답을 놓침",
        "threshold": "F1 ≥ 0.85 이면 양호",
    },
    "fpr_fnr": {
        "title": "FPR / FNR (오탐율 / 미탐율)",
        "desc": (
            "FPR(False Positive Rate): 인간이 '오답'이라 한 것을 judge가 '정답'으로 판정한 비율 — 과대평가 경향. "
            "FNR(False Negative Rate): 인간이 '정답'이라 한 것을 judge가 '오답'으로 판정한 비율 — 과소평가 경향."
        ),
        "high": "높을수록 나쁨 — 운영 시 품질 게이트가 무력화(FPR) 또는 좋은 답변을 거부(FNR)",
        "low": "낮을수록 좋음 — 0에 가까울수록 오판이 없음",
        "threshold": "FPR ≤ 0.10, FNR ≤ 0.10 이면 양호",
    },
    "score_gap": {
        "title": "Score Gap (Judge vs Human 점수 차이)",
        "desc": "Judge score(judge가 정답이라 한 비율)에서 Human score(실제 정답 비율)를 뺀 값입니다.",
        "high": "양수: judge가 lenient(너그럽게 평가) — 실제보다 높은 점수 부여",
        "low": "음수: judge가 strict(엄격하게 평가) — 실제보다 낮은 점수 부여. 0에 가까울수록 이상적",
        "threshold": "−0.05 ~ +0.05 이면 편향 없음으로 간주",
    },
    "prompt_sensitivity": {
        "title": "Prompt Sensitivity (프롬프트 민감도)",
        "desc": (
            "동일 샘플에 대해 프롬프트 템플릿(minimal / guideline / guideline_with_examples)을 바꿨을 때 "
            "레이블이 뒤집히는 비율(label_flip_rate)을 heatmap으로 표시합니다."
        ),
        "high": "높을수록 나쁨 — 프롬프트 문구에 따라 판단이 흔들리는 불안정한 judge",
        "low": "낮을수록 좋음 — 0에 가까울수록 프롬프트 변화에 강인한 judge",
        "threshold": "≤ 0.10 이면 안정적",
    },
    "dummy_robustness": {
        "title": "Dummy Answer Robustness (더미 응답 강건성)",
        "desc": (
            "gold_answer_verbatim(정답 그대로), Yes, Sure, 질문 반복, 빈 답변 등 "
            "더미 응답에 대해 올바르게 판단하는 정확도입니다."
        ),
        "high": "높을수록 좋음 — 1.0이면 모든 더미 케이스를 정확히 처리",
        "low": "낮을수록 나쁨 — 'Sure', 'Yes' 같은 무의미한 응답을 정답으로 오인",
        "threshold": "≥ 0.90 이면 양호",
    },
    "cost_latency": {
        "title": "Cost / Latency Trade-off",
        "desc": "Judge 모델의 평균 추론 지연(ms)과 총 추정 비용의 trade-off를 보여줍니다.",
        "high": "비용·지연이 낮으면서 성능(Scott's π)이 높은 우측 상단이 이상적",
        "low": "비용이 높고 성능이 낮으면 비효율적",
        "threshold": "운영 목적에 따라 허용 범위가 다름",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (EmptyDataError, FileNotFoundError):
        return pd.DataFrame()


def _fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def _metric_card(title: str, desc: str, high: str, low: str, threshold: str, img_b64: str) -> str:
    return f"""
    <section class="metric-card">
      <h2>{title}</h2>
      <p class="metric-desc">{desc}</p>
      <div class="metric-guide">
        <span class="badge badge-good">▲ 높을 때: {high}</span>
        <span class="badge badge-warn">▼ 낮을 때: {low}</span>
        <span class="badge badge-info">기준: {threshold}</span>
      </div>
      <div class="chart-wrap">
        <img src="data:image/png;base64,{img_b64}" alt="{title}" />
      </div>
    </section>
"""


def _html_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "<p class='no-data'>데이터 없음</p>"
    cols = frame.columns.tolist()
    header = "".join(f"<th>{c}</th>" for c in cols)
    rows_html = ""
    for row in frame.fillna("").astype(str).itertuples(index=False, name=None):
        rows_html += "<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>"
    return f"<table><thead><tr>{header}</tr></thead><tbody>{rows_html}</tbody></table>"


# ---------------------------------------------------------------------------
# Individual plot builders (return base64 strings)
# ---------------------------------------------------------------------------


def _plot_bar(labels: list[str], values: list[float], title: str, ylabel: str, color: str = "#4C8BF5") -> str:
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    bars = ax.bar(labels, values, color=color, width=0.6)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(1.05, max(values) * 1.15) if values else 1.05)
    ax.tick_params(axis="x", rotation=35)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    return _fig_to_base64(fig)


def _plot_grouped_bar(
    x_labels: list[str],
    series: dict[str, list[float]],
    title: str,
    colors: list[str] | None = None,
) -> str:
    n_series = len(series)
    n_groups = len(x_labels)
    width = 0.7 / n_series
    fig, ax = plt.subplots(figsize=(max(8, n_groups * 1.4), 5))
    default_colors = ["#4C8BF5", "#F5564C", "#F5A623", "#34A853"]
    for i, (name, vals) in enumerate(series.items()):
        offsets = [j + (i - n_series / 2 + 0.5) * width for j in range(n_groups)]
        c = (colors[i] if colors and i < len(colors) else default_colors[i % len(default_colors)])
        ax.bar(offsets, vals, width=width * 0.9, label=name, color=c)
    ax.set_xticks(list(range(n_groups)))
    ax.set_xticklabels(x_labels, rotation=35, ha="right")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.legend()
    fig.tight_layout()
    return _fig_to_base64(fig)


def _plot_heatmap(pivot: pd.DataFrame, title: str, cmap: str = "YlOrRd") -> str:
    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.8), max(4, len(pivot.index) * 1.2)))
    data = pivot.to_numpy(dtype=float)
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.tolist(), rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist())
    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            ax.text(c, r, f"{data[r, c]:.2f}", ha="center", va="center", fontsize=9, color="white" if data[r, c] > 0.5 else "black")
    fig.tight_layout()
    return _fig_to_base64(fig)


def _plot_scatter(x: list[float], y: list[float], labels: list[str], xlabel: str, ylabel: str, title: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, s=80, color="#4C8BF5", zorder=3)
    for xi, yi, lab in zip(x, y, labels):
        ax.annotate(lab, (xi, yi), textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _fig_to_base64(fig)


# ---------------------------------------------------------------------------
# Paper reference plots
# ---------------------------------------------------------------------------


def _build_paper_reference_plots() -> list[tuple[str, str]]:
    """Return list of (section_html, b64_img) for paper reference plots."""
    ref = PAPER_REFERENCE
    models = ref["model"]
    sections = []

    def ref_bar(key: str, ylabel: str, color: str, title: str) -> str:
        fig, ax = plt.subplots(figsize=(9, 4))
        bars = ax.bar(models, ref[key], color=color, width=0.6)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.1)
        ax.tick_params(axis="x", rotation=30)
        for bar, val in zip(bars, ref[key]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.2f}", ha="center", fontsize=8)
        fig.tight_layout()
        return _fig_to_base64(fig)

    sections.append(("Scott's π", ref_bar("scotts_pi", "Scott's pi", "#34A853", "Reference (Paper): Scott's pi on TriviaQA")))
    sections.append(("Percent Agreement", ref_bar("percent_agreement", "Agreement", "#4C8BF5", "Reference (Paper): Percent Agreement on TriviaQA")))

    # FPR + FNR grouped
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(models))
    w = 0.35
    ax.bar(x - w / 2, ref["fpr"], width=w, label="FPR", color="#F5564C")
    ax.bar(x + w / 2, ref["fnr"], width=w, label="FNR", color="#F5A623")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_title("Reference (Paper): FPR / FNR on TriviaQA", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 0.5)
    ax.legend()
    fig.tight_layout()
    sections.append(("FPR / FNR", _fig_to_base64(fig)))

    return sections


# ---------------------------------------------------------------------------
# Callout helpers
# ---------------------------------------------------------------------------


def _ranking_winner(rankings: pd.DataFrame, ranking_name: str) -> str:
    rows = rankings[rankings["ranking_name"] == ranking_name]
    if rows.empty:
        return "n/a"
    winner = rows.sort_values(["rank", "judge_model"]).iloc[0]
    prompt = f", prompt={winner['prompt_template']}" if winner.get("prompt_template") else ""
    return f"{winner['judge_model']}{prompt} ({winner['metric_value']:.3f})"


def _low_cost_callout(metrics: pd.DataFrame) -> str:
    if metrics.empty:
        return "n/a"
    frame = metrics.copy()
    frame["total_estimated_cost"] = frame["total_estimated_cost"].fillna(0.0)
    frame = frame.sort_values(["total_estimated_cost", "scotts_pi"], ascending=[True, False])
    row = frame.iloc[0]
    return f"{row['judge_model']}, prompt={row['prompt_template']} (cost={row['total_estimated_cost']:.3f}, Scott's π={row['scotts_pi']:.3f})"


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
    return f"{row['judge_model']} (accuracy={row['robustness_accuracy']:.3f})"


def _operational_readiness(metrics: pd.DataFrame, leniency: pd.DataFrame) -> str:
    if metrics.empty:
        return "<p>데이터 없음</p>"
    row = metrics.sort_values(["invalid_rate", "scotts_pi"], ascending=[True, False]).iloc[0]
    ready = row["invalid_rate"] < 0.05 and row["scotts_pi"] >= 0.5
    badge = "<span class='badge badge-good'>운영 투입 가능</span>" if ready else "<span class='badge badge-warn'>추가 검증 필요</span>"
    return (
        f"<p>추천 후보: <strong>{row['judge_model']}</strong> (prompt={row['prompt_template']}) {badge}</p>"
        f"<ul>"
        f"<li>Invalid rate: {row['invalid_rate']:.3f}</li>"
        f"<li>Scott's π: {row['scotts_pi']:.3f}</li>"
        f"<li>Score delta: {row['score_delta']:.3f}</li>"
        f"</ul>"
    )


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: #f0f2f5;
  color: #1a1a2e;
  line-height: 1.6;
}
header {
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
  color: white;
  padding: 2.5rem 2rem 2rem;
  box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
header h1 { font-size: 2rem; font-weight: 700; letter-spacing: -0.5px; }
header p { opacity: 0.75; margin-top: 0.4rem; font-size: 0.95rem; }
.container { max-width: 1200px; margin: 0 auto; padding: 2rem 1.5rem; }
.summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 1rem;
  margin-bottom: 2.5rem;
}
.summary-item {
  background: white;
  border-radius: 12px;
  padding: 1.2rem 1.4rem;
  box-shadow: 0 2px 8px rgba(0,0,0,0.07);
  border-left: 4px solid #4C8BF5;
}
.summary-item .label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; color: #666; }
.summary-item .value { font-size: 1rem; font-weight: 600; margin-top: 0.3rem; color: #1a1a2e; }
.metric-card {
  background: white;
  border-radius: 14px;
  padding: 1.8rem 2rem;
  margin-bottom: 2rem;
  box-shadow: 0 2px 10px rgba(0,0,0,0.07);
}
.metric-card h2 { font-size: 1.25rem; font-weight: 700; margin-bottom: 0.6rem; color: #1a1a2e; }
.metric-desc { color: #444; font-size: 0.95rem; margin-bottom: 0.8rem; }
.metric-guide { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1.2rem; }
.badge {
  display: inline-block;
  padding: 0.25rem 0.65rem;
  border-radius: 20px;
  font-size: 0.78rem;
  font-weight: 500;
}
.badge-good { background: #d4edda; color: #155724; }
.badge-warn { background: #fff3cd; color: #856404; }
.badge-info { background: #d1ecf1; color: #0c5460; }
.badge-ref { background: #e2d9f3; color: #4a1d96; }
.chart-wrap { text-align: center; }
.chart-wrap img { max-width: 100%; border-radius: 8px; border: 1px solid #eee; }
section.section-block {
  background: white;
  border-radius: 14px;
  padding: 1.8rem 2rem;
  margin-bottom: 2rem;
  box-shadow: 0 2px 10px rgba(0,0,0,0.07);
}
section.section-block h2 { font-size: 1.25rem; font-weight: 700; margin-bottom: 1rem; }
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.82rem;
  margin-top: 0.5rem;
  overflow-x: auto;
  display: block;
}
th { background: #f0f4ff; padding: 0.55rem 0.75rem; text-align: left; font-weight: 600; border-bottom: 2px solid #d0d8ef; }
td { padding: 0.45rem 0.75rem; border-bottom: 1px solid #f0f0f0; }
tr:hover td { background: #f9fbff; }
.no-data { color: #888; font-style: italic; }
.ref-note {
  background: #f8f4ff;
  border-left: 4px solid #7c3aed;
  padding: 0.9rem 1.2rem;
  border-radius: 0 8px 8px 0;
  margin-bottom: 1.5rem;
  font-size: 0.88rem;
  color: #4a1d96;
}
.ref-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 1.5rem; }
.ref-plot-item { background: #fdf9ff; border-radius: 10px; padding: 1rem; border: 1px solid #e4d8f7; }
.ref-plot-item h3 { font-size: 0.95rem; font-weight: 600; margin-bottom: 0.6rem; color: #5b21b6; }
.operational-box {
  background: #f0f9ff;
  border-radius: 10px;
  padding: 1.2rem 1.5rem;
  border: 1px solid #bae6fd;
}
.operational-box ul { margin-top: 0.5rem; padding-left: 1.5rem; }
.operational-box li { margin-top: 0.3rem; font-size: 0.9rem; }
footer {
  text-align: center;
  padding: 2rem;
  color: #888;
  font-size: 0.82rem;
  border-top: 1px solid #e0e0e0;
  margin-top: 2rem;
}
h1.section-title {
  font-size: 1.5rem;
  font-weight: 700;
  margin: 2.5rem 0 1.2rem;
  color: #1a1a2e;
  border-bottom: 2px solid #e2e8f0;
  padding-bottom: 0.5rem;
}
"""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_report(output_dir: Path) -> Path:
    metrics = _safe_read_csv(output_dir / "metrics_overall.csv")
    rankings = _safe_read_csv(output_dir / "model_rankings.csv")
    prompt_sensitivity = _safe_read_csv(output_dir / "prompt_sensitivity.csv")
    reference_order = _safe_read_csv(output_dir / "reference_order_sensitivity.csv")
    dummy_robustness = _safe_read_csv(output_dir / "dummy_answer_robustness.csv")
    answer_source_metrics = _safe_read_csv(output_dir / "metrics_by_answer_source.csv")
    leniency = _safe_read_csv(output_dir / "leniency_bias.csv")

    # ------------------------------------------------------------------
    # Build all plots as base64
    # ------------------------------------------------------------------
    plots: dict[str, str] = {}

    if not metrics.empty:
        labels = (metrics["judge_model"] + ":" + metrics["prompt_template"]).tolist()

        plots["scotts_pi"] = _plot_bar(labels, metrics["scotts_pi"].tolist(), "Scott's π by Judge / Prompt", "Scott's π", "#34A853")
        plots["percent_agreement"] = _plot_bar(labels, metrics["percent_agreement"].tolist(), "Percent Agreement by Judge / Prompt", "Agreement", "#4C8BF5")
        plots["score_gap"] = _plot_bar(
            labels,
            metrics["score_delta"].tolist(),
            "Score Gap (Judge − Human) by Judge / Prompt",
            "Score delta",
            ["#F5564C" if v > 0 else "#34A853" for v in metrics["score_delta"].tolist()],  # type: ignore[arg-type]
        )

        n = len(metrics)
        plots["judge_vs_human"] = _plot_scatter(
            metrics["human_score"].tolist(),
            metrics["judge_score"].tolist(),
            labels,
            "Human Score",
            "Judge Score",
            "Judge Score vs Human Score",
        )
        plots["precision_recall_f1"] = _plot_grouped_bar(
            labels,
            {"Precision": metrics["precision"].tolist(), "Recall": metrics["recall"].tolist(), "F1": metrics["f1"].tolist()},
            "Precision / Recall / F1",
        )
        plots["fpr_fnr"] = _plot_grouped_bar(
            labels,
            {"FPR": metrics["fpr"].tolist(), "FNR": metrics["fnr"].tolist()},
            "False Positive / False Negative Rates",
            colors=["#F5564C", "#F5A623"],
        )

        has_latency = "avg_latency_ms" in metrics.columns
        has_cost = "total_estimated_cost" in metrics.columns
        lat = metrics["avg_latency_ms"].fillna(0).tolist() if has_latency else [0] * n
        cost = metrics["total_estimated_cost"].fillna(0).tolist() if has_cost else [0] * n
        plots["cost_latency"] = _plot_scatter(lat, cost, labels, "Avg Latency (ms)", "Estimated Cost", "Cost / Latency Trade-off")

    if not prompt_sensitivity.empty:
        pivot = prompt_sensitivity.pivot_table(
            index="judge_model",
            columns="prompt_left",
            values="label_flip_rate",
            aggfunc="mean",
            fill_value=0.0,
        )
        plots["prompt_sensitivity_heatmap"] = _plot_heatmap(pivot, "Prompt Sensitivity — Label Flip Rate", cmap="YlOrRd")

    if not answer_source_metrics.empty:
        pivot2 = answer_source_metrics.pivot_table(
            index="judge_model",
            columns="answer_source",
            values="scotts_pi",
            aggfunc="mean",
            fill_value=0.0,
        )
        plots["answer_source_heatmap"] = _plot_heatmap(pivot2, "Scott's π by Answer Source", cmap="Blues")

    if not dummy_robustness.empty:
        dr_labels = (dummy_robustness["judge_model"] + ":" + dummy_robustness["dummy_class"]).tolist()
        plots["dummy_robustness"] = _plot_bar(dr_labels, dummy_robustness["robustness_accuracy"].tolist(), "Dummy Answer Robustness", "Accuracy", "#7C3AED")

    paper_ref_plots = _build_paper_reference_plots()

    # ------------------------------------------------------------------
    # Collect experiment metadata
    # ------------------------------------------------------------------
    config_path = output_dir / "config.resolved.yaml"
    exp_name = output_dir.name
    try:
        import yaml
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        exp_name = cfg.get("experiment_name", exp_name)
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Build callouts
    # ------------------------------------------------------------------
    callout_items = [
        ("최우수 judge (Scott's π)", _ranking_winner(rankings, "primary_scotts_pi")),
        ("엄격 게이트 최우수 (FPR 최소)", _ranking_winner(rankings, "false_positive_rate")),
        ("최저 비용 judge", _low_cost_callout(metrics)),
        ("가장 관대한 judge", _leniency_callout(leniency, most_lenient=True)),
        ("가장 엄격한 judge", _leniency_callout(leniency, most_lenient=False)),
        ("프롬프트 민감도 최고", _prompt_callout(prompt_sensitivity)),
        ("더미 강건성 최저", _dummy_callout(dummy_robustness)),
    ]
    callout_html = "".join(
        f'<div class="summary-item"><div class="label">{label}</div><div class="value">{value}</div></div>'
        for label, value in callout_items
    )

    # ------------------------------------------------------------------
    # Assemble HTML
    # ------------------------------------------------------------------
    def section(title: str, content: str) -> str:
        return f'<section class="section-block"><h2>{title}</h2>{content}</section>'

    def metric_section(key: str, plot_key: str) -> str:
        m = METRIC_DESCRIPTIONS[key]
        img = plots.get(plot_key, "")
        img_tag = f'<div class="chart-wrap"><img src="data:image/png;base64,{img}" alt="{m["title"]}" /></div>' if img else "<p class='no-data'>차트 없음</p>"
        return f"""
        <section class="metric-card">
          <h2>{m['title']}</h2>
          <p class="metric-desc">{m['desc']}</p>
          <div class="metric-guide">
            <span class="badge badge-good">▲ {m['high']}</span>
            <span class="badge badge-warn">▼ {m['low']}</span>
            <span class="badge badge-info">기준: {m['threshold']}</span>
          </div>
          {img_tag}
        </section>
"""

    body_parts = [
        f'<div class="summary-grid">{callout_html}</div>',
        '<h1 class="section-title">핵심 성능 지표</h1>',
        metric_section("scotts_pi", "scotts_pi"),
        metric_section("percent_agreement", "percent_agreement"),
        metric_section("precision_recall_f1", "precision_recall_f1"),
        metric_section("fpr_fnr", "fpr_fnr"),
        metric_section("score_gap", "score_gap"),
        metric_section("cost_latency", "cost_latency"),
        '<h1 class="section-title">프롬프트 민감도 분석</h1>',
        metric_section("prompt_sensitivity", "prompt_sensitivity_heatmap"),
        section(
            "프롬프트별 상세 비교",
            _html_table(
                prompt_sensitivity[["judge_model", "prompt_left", "prompt_right", "label_flip_rate", "scotts_pi_by_prompt", "prompt_consistency"]].round(4)
                if not prompt_sensitivity.empty
                else pd.DataFrame()
            ),
        ),
        '<h1 class="section-title">Reference Order 민감도 분석</h1>',
        section(
            "Reference Order 적용 범위",
            _html_table(
                reference_order[
                    [
                        "judge_model",
                        "reference_order_consistency",
                        "label_flip_rate_by_reference_order",
                        "eligible_sample_groups",
                        "skipped_single_alias_groups",
                        "coverage",
                        "valid_rows",
                    ]
                ].round(4)
                if not reference_order.empty
                else pd.DataFrame()
            ),
        ),
        section(
            "Reference Order 해석 가이드",
            (
                "<p>eligible_sample_groups는 alias가 2개 이상이라 순서 변경 테스트가 가능했던 샘플 수입니다. "
                "skipped_single_alias_groups는 alias가 1개라 테스트가 적용되지 않은 샘플 수입니다. "
                "coverage는 전체 base 샘플 중 실제 reference-order test에 포함된 비율입니다.</p>"
            ),
        ),
        '<h1 class="section-title">Answer Source별 분석</h1>',
        section(
            "Answer Source별 Scott's π",
            (
                f'<div class="chart-wrap"><img src="data:image/png;base64,{plots["answer_source_heatmap"]}" /></div>'
                if "answer_source_heatmap" in plots
                else "<p class='no-data'>데이터 없음</p>"
            ),
        ),
        '<h1 class="section-title">더미 응답 강건성</h1>',
        metric_section("dummy_robustness", "dummy_robustness"),
        section(
            "더미 응답 상세",
            _html_table(dummy_robustness) if not dummy_robustness.empty else pd.DataFrame(),
        ) if not dummy_robustness.empty else "",
        '<h1 class="section-title">Judge vs Human Score</h1>',
        section(
            "Judge Score vs Human Score",
            (
                f'<div class="chart-wrap"><img src="data:image/png;base64,{plots["judge_vs_human"]}" /></div>'
                if "judge_vs_human" in plots
                else "<p class='no-data'>데이터 없음</p>"
            ),
        ),
        '<h1 class="section-title">운영 투입 판단</h1>',
        section("Operational Readiness", f'<div class="operational-box">{_operational_readiness(metrics, leniency)}</div>'),
        '<h1 class="section-title">전체 메트릭 테이블</h1>',
        section(
            "Metrics Overview",
            _html_table(
                metrics[["judge_model", "prompt_template", "scotts_pi", "percent_agreement", "precision", "recall", "f1", "fpr", "fnr", "score_delta", "invalid_rate"]].round(4)
                if not metrics.empty
                else pd.DataFrame()
            ),
        ),
        section("Model Rankings", _html_table(rankings) if not rankings.empty else pd.DataFrame()),
        # ------------------------------------------------------------------
        # Paper reference section
        # ------------------------------------------------------------------
        '<h1 class="section-title">논문 비교 참고 (별도 데이터셋)</h1>',
        f"""
        <section class="section-block">
          <div class="ref-note">
            <strong>⚠ 참고용 데이터</strong> — 아래 수치는 Bavaresco et al., 2024
            <em>"Judging the Judges: Evaluating Alignment and Vulnerabilities in LLMs-as-Judges"</em>
            (arXiv:2406.12624)의 TriviaQA 실험 결과를 기반으로 합니다.
            본 실험과 <strong>데이터셋이 다르므로 수치 직접 비교는 지양</strong>하고,
            judge 모델 군 간 상대적 순위와 특성 파악의 참고로만 활용하세요.
          </div>
          <div class="ref-grid">
            {"".join(f'<div class="ref-plot-item"><h3>{name}</h3><img src="data:image/png;base64,{b64}" style="width:100%;border-radius:6px;" /></div>' for name, b64 in paper_ref_plots)}
          </div>
        </section>
        """,
    ]

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Judge Evaluation Report — {exp_name}</title>
  <style>{_CSS}</style>
</head>
<body>
  <header>
    <h1>Judge Model Evaluation Report</h1>
    <p>실험: <strong>{exp_name}</strong> &nbsp;|&nbsp; 출력 디렉토리: {output_dir}</p>
  </header>
  <div class="container">
    {"".join(body_parts)}
  </div>
  <footer>
    Generated by judge-model-evaluation &nbsp;·&nbsp;
    Bavaresco et al. 2024 reference data: arXiv:2406.12624
  </footer>
</body>
</html>
"""

    report_path = output_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path
