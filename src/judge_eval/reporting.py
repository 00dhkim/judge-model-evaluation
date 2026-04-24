from __future__ import annotations

import base64
import io
from pathlib import Path
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from judge_eval.utils import ensure_dir

matplotlib.use("Agg")

# Korean font setup — register local user fonts first so matplotlib can render Hangul in saved PNGs.
import matplotlib.font_manager as _fm


def _configure_korean_font() -> None:
    candidate_paths = [
        Path.home() / ".local/share/fonts/malgun.ttf",
        Path.home() / ".local/share/fonts/malgunbd.ttf",
        Path.home() / ".local/share/fonts/NanumGothic.ttf",
    ]
    for path in candidate_paths:
        if path.exists():
            try:
                _fm.fontManager.addfont(str(path))
            except RuntimeError:
                pass

    preferred_names = [
        "Malgun Gothic",
        "맑은 고딕",
        "NanumGothic",
        "나눔고딕",
        "Apple SD Gothic Neo",
        "Noto Sans KR",
        "Noto Sans CJK KR",
        "Noto Serif CJK KR",
    ]
    available_names = {f.name for f in _fm.fontManager.ttflist}
    korean_font_name = next((name for name in preferred_names if name in available_names), None)

    if korean_font_name:
        matplotlib.rcParams["font.family"] = [korean_font_name, "DejaVu Sans", "sans-serif"]
        matplotlib.rcParams["font.sans-serif"] = [korean_font_name, "DejaVu Sans"]
        return

    warnings.warn(
        "No Korean font found for matplotlib. Install or register a Hangul-capable font such as "
        "Malgun Gothic or NanumGothic to avoid missing glyph warnings.",
        UserWarning,
        stacklevel=2,
    )


_configure_korean_font()
matplotlib.rcParams["axes.unicode_minus"] = False

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


def _safe_read_parquet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except (FileNotFoundError, OSError, ValueError, ImportError):
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


def _format_int(value: int | float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{int(value):,}"


def _format_pct(value: float | None, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.{digits}%}"


def _format_ms(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:,.0f} ms"


def _dataset_summary(dataset_meta: dict, normalized_samples: pd.DataFrame, metrics_by_dataset: pd.DataFrame) -> str:
    dataset_names: list[str] = []
    if "dataset" in normalized_samples.columns and not normalized_samples.empty:
        dataset_names = sorted(normalized_samples["dataset"].dropna().astype(str).unique().tolist())
    elif "dataset" in metrics_by_dataset.columns and not metrics_by_dataset.empty:
        dataset_names = sorted(metrics_by_dataset["dataset"].dropna().astype(str).unique().tolist())

    if not dataset_names:
        return "n/a"
    return f"{len(dataset_names)}개 ({', '.join(dataset_names)})"


def _answer_source_summary(dataset_meta: dict, normalized_samples: pd.DataFrame) -> str:
    sources = dataset_meta.get("answer_sources")
    if isinstance(sources, list) and sources:
        source_names = [str(source) for source in sources]
    elif "answer_source" in normalized_samples.columns and not normalized_samples.empty:
        source_names = sorted(normalized_samples["answer_source"].dropna().astype(str).unique().tolist())
    else:
        source_names = []

    if not source_names:
        return "n/a"
    return f"{len(source_names)}개 ({', '.join(source_names)})"


def _experiment_overview_items(
    dataset_meta: dict,
    normalized_samples: pd.DataFrame,
    parsed_predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    metrics_by_dataset: pd.DataFrame,
) -> list[tuple[str, str]]:
    base_predictions = (
        parsed_predictions.loc[parsed_predictions["variant_type"].eq("base")].copy()
        if "variant_type" in parsed_predictions.columns
        else parsed_predictions.copy()
    )

    total_samples = None
    if isinstance(dataset_meta.get("row_count"), (int, float)):
        total_samples = int(dataset_meta["row_count"])
    elif not normalized_samples.empty:
        total_samples = len(normalized_samples)

    question_count = None
    if "question" in normalized_samples.columns and not normalized_samples.empty:
        question_count = normalized_samples["question"].dropna().nunique()
    elif "sample_id" in normalized_samples.columns and not normalized_samples.empty:
        question_count = normalized_samples["sample_id"].dropna().nunique()

    judge_config_count = None
    if {"judge_model", "prompt_template"}.issubset(metrics.columns) and not metrics.empty:
        judge_config_count = metrics[["judge_model", "prompt_template"]].drop_duplicates().shape[0]

    human_positive_rate = None
    if "human_label" in normalized_samples.columns and not normalized_samples.empty:
        human_positive_rate = normalized_samples["human_label"].astype(bool).mean()

    valid_rate = None
    if "parse_status" in base_predictions.columns and not base_predictions.empty:
        valid_rate = base_predictions["parse_status"].eq("ok").mean()
    elif "valid_rate" in metrics.columns and not metrics.empty:
        valid_rate = metrics["valid_rate"].mean()

    p50_latency = None
    p95_latency = None
    if "latency_ms" in base_predictions.columns and not base_predictions.empty:
        p50_latency = float(base_predictions["latency_ms"].dropna().quantile(0.50))
        p95_latency = float(base_predictions["latency_ms"].dropna().quantile(0.95))
    elif {"p50_latency_ms", "p95_latency_ms"}.issubset(metrics.columns) and not metrics.empty:
        p50_latency = float(metrics["p50_latency_ms"].dropna().median())
        p95_latency = float(metrics["p95_latency_ms"].dropna().median())

    return [
        ("총 평가 샘플", _format_int(total_samples)),
        ("고유 질문 수", _format_int(question_count)),
        ("답변 소스 구성", _answer_source_summary(dataset_meta, normalized_samples)),
        ("데이터셋 구성", _dataset_summary(dataset_meta, normalized_samples, metrics_by_dataset)),
        ("평가 judge 설정 수", _format_int(judge_config_count)),
        ("인간 기준 정답 비율", _format_pct(human_positive_rate)),
        ("유효 판정률(base)", _format_pct(valid_rate)),
        ("지연 시간(base)", f"p50 {_format_ms(p50_latency)} · p95 {_format_ms(p95_latency)}"),
    ]


# ---------------------------------------------------------------------------
# Individual plot builders (return base64 strings)
# ---------------------------------------------------------------------------


def _plot_bar(labels: list[str], values: list[float], title: str, ylabel: str, color: str = "#4C8BF5") -> str:
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    bars = ax.bar(labels, values, color=color, width=0.6)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.0)
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
    ax.set_ylim(0, 1.0)
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


def _plot_model_rankings_chart(rankings: pd.DataFrame) -> str:
    """Rank heatmap for model rankings — standard academic leaderboard style.

    Rows = model+prompt configs, columns = ranking criteria.
    Cell color encodes rank (green=1st, red=last); value is annotated inside.
    Avoids the false 'connectivity' implied by bump/line charts for independent metrics.
    """
    if rankings.empty:
        return ""

    METRIC_SHORT = {
        "primary_scotts_pi": "Scott's π",
        "percent_agreement": "% Agree",
        "absolute_score_gap": "Score Gap↓",
        "false_positive_rate": "FPR↓",
        "false_negative_rate": "FNR↓",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
    }
    # Lower-is-better metrics get inverted color scale
    LOWER_BETTER = {"absolute_score_gap", "false_positive_rate", "false_negative_rate"}

    rankings = rankings.copy()
    rankings["model_key"] = rankings["judge_model"] + "\n(" + rankings["prompt_template"] + ")"

    # Build rank pivot and value pivot
    rank_pivot = rankings.pivot_table(index="model_key", columns="ranking_name", values="rank", aggfunc="min")
    val_pivot = rankings.pivot_table(index="model_key", columns="ranking_name", values="metric_value", aggfunc="mean")

    # Order columns by METRIC_SHORT key order
    col_order = [c for c in METRIC_SHORT if c in rank_pivot.columns] + [c for c in rank_pivot.columns if c not in METRIC_SHORT]
    rank_pivot = rank_pivot.reindex(columns=col_order)
    val_pivot = val_pivot.reindex(columns=col_order)

    # Sort rows: models that are rank-1 most often go first (lower avg rank = better)
    rank_pivot["_avg"] = rank_pivot.mean(axis=1)
    rank_pivot = rank_pivot.sort_values("_avg").drop(columns="_avg")
    val_pivot = val_pivot.reindex(index=rank_pivot.index)

    n_rows, n_cols = rank_pivot.shape
    cell_w, cell_h = 1.1, 0.85
    fig_w = min(max(8, n_cols * cell_w + 2.5), 13)  # cap at 13 inches
    fig_h = max(4, n_rows * cell_h + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    max_rank = int(rank_pivot.values[~np.isnan(rank_pivot.values)].max()) if n_rows > 0 else 1

    for ri, model in enumerate(rank_pivot.index):
        for ci, col in enumerate(rank_pivot.columns):
            rank_val = rank_pivot.loc[model, col]
            metric_val = val_pivot.loc[model, col]
            if pd.isna(rank_val):
                ax.add_patch(plt.Rectangle((ci, ri), 1, 1, color="#F0F0F0", zorder=1))
                ax.text(ci + 0.5, ri + 0.5, "—", ha="center", va="center", fontsize=9, color="#AAA")
                continue

            # Normalize rank to [0,1] for colormap (0=best color)
            norm = (rank_val - 1) / max(max_rank - 1, 1)
            if col in LOWER_BETTER:
                # For lower-is-better: rank 1 (lowest value) is still "best"
                # color stays: norm=0 → green, norm=1 → red
                pass
            color = plt.cm.Blues(0.85 - norm * 0.65)  # type: ignore[attr-defined]  # rank1=dark, last=light

            ax.add_patch(plt.Rectangle((ci, ri), 1, 1, color=color, zorder=1, linewidth=0))

            # Rank number (large, bold)
            text_color = "white" if norm < 0.15 else "#222"  # only rank 1 (darkest) gets white
            ax.text(ci + 0.5, ri + 0.63, f"#{int(rank_val)}", ha="center", va="center",
                    fontsize=14, fontweight="bold", color=text_color, zorder=2)
            # Metric value (below rank number)
            val_str = f"{metric_val:.3f}" if not pd.isna(metric_val) else ""
            ax.text(ci + 0.5, ri + 0.24, val_str, ha="center", va="center",
                    fontsize=10, color=text_color, alpha=0.9, zorder=2)

    # Grid lines
    for x in range(n_cols + 1):
        ax.axvline(x, color="white", lw=2, zorder=3)
    for y in range(n_rows + 1):
        ax.axhline(y, color="white", lw=2, zorder=3)

    # Axis labels
    col_labels = [METRIC_SHORT.get(c, c) for c in rank_pivot.columns]
    ax.set_xticks([i + 0.5 for i in range(n_cols)])
    ax.set_xticklabels(col_labels, fontsize=10, fontweight="600")
    ax.set_yticks([i + 0.5 for i in range(n_rows)])
    ax.set_yticklabels(rank_pivot.index.tolist(), fontsize=9)
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.tick_params(length=0)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    for spine in ax.spines.values():
        spine.set_visible(False)

    # Colorbar legend
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    sm = plt.cm.ScalarMappable(cmap="Blues", norm=mcolors.Normalize(vmin=1, vmax=max_rank))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, aspect=30)
    cb.set_label("Rank", fontsize=9)
    cb.ax.invert_yaxis()  # rank 1 at top = green

    fig.suptitle("Model Rankings — Cross-Metric Overview", fontsize=13, fontweight="bold", y=1.02)
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
# Reference Order visual (3-panel per model)
# ---------------------------------------------------------------------------


def _plot_reference_order_visual(df: pd.DataFrame) -> str:
    """Tri-panel visual: gauge + donut + stacked bar for reference order sensitivity."""
    from matplotlib.patches import Wedge

    n = len(df)
    fig, all_axes = plt.subplots(n, 3, figsize=(14, 5 * n))
    if n == 1:
        all_axes = [all_axes]

    def v2a(v: float) -> float:
        return 180.0 - v * 180.0

    def metric_or_default(value: object, default: float = 0.0) -> float:
        if pd.isna(value):
            return default
        return float(value)

    def draw_gauge(ax: plt.Axes, value: float, title: str, good: float = 0.9, warn: float = 0.8) -> None:
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.2, 0.85)
        cx, cy = 0.5, 0.35
        r_o, r_i = 0.38, 0.22
        for sv, ev, col in [(0.0, warn, "#F5564C"), (warn, good, "#F5A623"), (good, 1.0, "#34A853")]:
            ax.add_patch(Wedge((cx, cy), r_o, v2a(ev), v2a(sv), width=r_o - r_i, color=col, alpha=0.9, zorder=1))
        # Tick marks at thresholds
        for tick_v, label in [(warn, f"{warn:.0%}"), (good, f"{good:.0%}")]:
            ta = np.radians(v2a(tick_v))
            ax.plot([cx + r_i * np.cos(ta), cx + (r_i - 0.04) * np.cos(ta)],
                    [cy + r_i * np.sin(ta), cy + (r_i - 0.04) * np.sin(ta)], color="#555", lw=1.5)
            ax.text(cx + (r_i - 0.1) * np.cos(ta), cy + (r_i - 0.1) * np.sin(ta),
                    label, ha="center", fontsize=7, color="#555")
        # Needle
        ang = np.radians(v2a(value))
        ax.annotate("", xy=(cx + (r_o - 0.02) * np.cos(ang), cy + (r_o - 0.02) * np.sin(ang)),
                    xytext=(cx, cy), arrowprops=dict(arrowstyle="->", color="#222", lw=2.5), zorder=3)
        ax.plot(cx, cy, "o", color="#222", markersize=5, zorder=4)
        # Value + status
        ax.text(cx, cy - 0.17, f"{value:.1%}", ha="center", fontsize=16, fontweight="bold")
        if value >= good:
            sc, st = "#34A853", "양호"
        elif value >= warn:
            sc, st = "#F5A623", "주의"
        else:
            sc, st = "#F5564C", "불량"
        ax.text(cx, -0.14, st, ha="center", fontsize=11, fontweight="bold", color=sc,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=sc, alpha=0.15, edgecolor=sc))
        ax.set_title(title, fontsize=11, fontweight="bold", pad=6)

    for ax_row, (_, row) in zip(all_axes, df.iterrows()):
        consistency = metric_or_default(row["reference_order_consistency"])
        flip_rate = metric_or_default(row["label_flip_rate_by_reference_order"])
        eligible = int(row["eligible_sample_groups"])
        skipped = int(row["skipped_single_alias_groups"])
        coverage = metric_or_default(row["coverage"])
        judge = str(row["judge_model"])
        n_flipped = round(eligible * flip_rate)
        n_consistent = eligible - n_flipped

        ax1, ax2, ax3 = ax_row[0], ax_row[1], ax_row[2]

        # Panel 1: Semicircle gauge for consistency rate
        draw_gauge(ax1, consistency, f"순서 일관성\n{judge}")

        # Panel 2: Donut — eligible vs skipped breakdown
        wedges, texts, autotexts = ax2.pie(
            [eligible, skipped],
            labels=[f"테스트 가능\n({eligible}개, alias≥2)", f"제외됨\n({skipped}개, alias=1)"],
            colors=["#4C8BF5", "#D1D5DB"],
            autopct="%1.0f%%",
            startangle=90,
            wedgeprops={"width": 0.5, "edgecolor": "white", "linewidth": 2},
            textprops={"fontsize": 9},
        )
        for at in autotexts:
            at.set_fontsize(9)
            at.set_fontweight("bold")
        ax2.set_title(f"샘플 적용 범위\n(전체 커버리지 {coverage:.0%})", fontsize=11, fontweight="bold")

        # Panel 3: Stacked bar — consistent vs flipped within eligible
        bar_h = 0.35
        ax3.barh([""], [n_consistent], color="#34A853", height=bar_h, label=f"일관됨 ({n_consistent}개)")
        ax3.barh([""], [n_flipped], left=[n_consistent], color="#F5564C", height=bar_h, label=f"순서 뒤집힘 ({n_flipped}개)")
        # 10% threshold line
        threshold_n = eligible * 0.10
        ax3.axvline(x=threshold_n, color="#F5A623", linestyle="--", lw=1.5, alpha=0.8, label=f"기준 10% ({threshold_n:.1f}개)")
        ax3.set_xlim(0, max(eligible, 1))
        ax3.set_xlabel("샘플 수", fontsize=9)
        ax3.set_title(f"라벨 플립 현황\n플립율 {flip_rate:.1%}  (테스트 가능 {eligible}개 기준)", fontsize=11, fontweight="bold")
        ax3.legend(loc="lower right", fontsize=8)
        ax3.set_yticks([])
        for spine in ["top", "right", "left"]:
            ax3.spines[spine].set_visible(False)
        if n_consistent > 0:
            ax3.text(n_consistent / 2, 0, str(n_consistent), ha="center", va="center", fontsize=13, fontweight="bold", color="white")
        if n_flipped > 0:
            ax3.text(n_consistent + n_flipped / 2, 0, str(n_flipped), ha="center", va="center", fontsize=13, fontweight="bold", color="white")

    fig.tight_layout(pad=2.5)
    return _fig_to_base64(fig)


def _has_reference_order_visual_data(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    required_columns = {"reference_order_consistency", "label_flip_rate_by_reference_order"}
    if not required_columns.issubset(df.columns):
        return False
    metrics = df[list(required_columns)]
    return bool(metrics.notna().any(axis=1).any())


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
        ax.set_ylim(0, 1.0)
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
    ax.set_ylim(0, 1.0)
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
  font-family: "Apple SD Gothic Neo", "Malgun Gothic", "Noto Sans KR", "NanumGothic",
    "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  background: #f0f2f5;
  color: #1a1a2e;
  line-height: 1.6;
  word-break: keep-all;
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
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


_EXP_COLORS = ["#4C8BF5", "#F5564C", "#F5A623", "#34A853", "#7C3AED", "#0891B2", "#DB2777", "#D97706", "#059669", "#DC2626"]


def _exp_color_map(exp_names: list[str]) -> dict[str, str]:
    return {name: _EXP_COLORS[i % len(_EXP_COLORS)] for i, name in enumerate(sorted(set(exp_names)))}


def generate_merged_report(output_dirs: list[Path], report_path: Path) -> Path:
    import yaml

    all_metrics: list[pd.DataFrame] = []
    all_prompt_sensitivity: list[pd.DataFrame] = []
    all_dummy_robustness: list[pd.DataFrame] = []
    all_rankings: list[pd.DataFrame] = []
    exp_names: list[str] = []

    for output_dir in output_dirs:
        metrics = _safe_read_csv(output_dir / "metrics_overall.csv")
        if metrics.empty:
            continue
        exp_name = output_dir.name
        config_path = output_dir / "config.resolved.yaml"
        if config_path.exists():
            try:
                cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                exp_name = cfg.get("experiment_name", exp_name)
            except Exception:
                pass

        metrics["experiment"] = exp_name
        all_metrics.append(metrics)
        exp_names.append(exp_name)

        ps = _safe_read_csv(output_dir / "prompt_sensitivity.csv")
        if not ps.empty:
            ps["experiment"] = exp_name
            all_prompt_sensitivity.append(ps)

        dr = _safe_read_csv(output_dir / "dummy_answer_robustness.csv")
        if not dr.empty:
            dr["experiment"] = exp_name
            all_dummy_robustness.append(dr)

        rankings = _safe_read_csv(output_dir / "model_rankings.csv")
        if not rankings.empty:
            rankings["experiment"] = exp_name
            all_rankings.append(rankings)

    if not all_metrics:
        raise ValueError("No metrics found in provided output directories")

    merged_metrics = pd.concat(all_metrics, ignore_index=True)
    merged_ps = pd.concat(all_prompt_sensitivity, ignore_index=True) if all_prompt_sensitivity else pd.DataFrame()
    merged_dr = pd.concat(all_dummy_robustness, ignore_index=True) if all_dummy_robustness else pd.DataFrame()
    merged_rankings = pd.concat(all_rankings, ignore_index=True) if all_rankings else pd.DataFrame()

    color_map = _exp_color_map(exp_names)
    best_per_model = (
        merged_metrics.sort_values("scotts_pi", ascending=False)
        .drop_duplicates("judge_model")
        .sort_values("scotts_pi", ascending=False)
        .reset_index(drop=True)
    )

    plots: dict[str, str] = {}

    labels_best = (best_per_model["judge_model"] + "\n(" + best_per_model["prompt_template"] + ")").tolist()
    bar_colors = [color_map.get(exp, "#4C8BF5") for exp in best_per_model["experiment"].tolist()]

    plots["scotts_pi"] = _plot_bar(labels_best, best_per_model["scotts_pi"].tolist(), "Scott's π — 모델별 최고 성능", "Scott's π", bar_colors)  # type: ignore[arg-type]
    plots["percent_agreement"] = _plot_bar(labels_best, best_per_model["percent_agreement"].tolist(), "Percent Agreement — 모델별 최고 성능", "Agreement", bar_colors)  # type: ignore[arg-type]
    plots["f1"] = _plot_bar(labels_best, best_per_model["f1"].tolist(), "F1 — 모델별 최고 성능", "F1", bar_colors)  # type: ignore[arg-type]
    plots["fpr_fnr"] = _plot_grouped_bar(
        labels_best,
        {"FPR": best_per_model["fpr"].tolist(), "FNR": best_per_model["fnr"].tolist()},
        "FPR / FNR — 모델별 최고 성능",
        colors=["#F5564C", "#F5A623"],
    )

    has_latency = "avg_latency_ms" in merged_metrics.columns
    has_cost = "total_estimated_cost" in merged_metrics.columns
    all_labels = (merged_metrics["judge_model"] + ":" + merged_metrics["prompt_template"]).tolist()
    lat = merged_metrics["avg_latency_ms"].fillna(0).tolist() if has_latency else [0] * len(merged_metrics)
    cost = merged_metrics["total_estimated_cost"].fillna(0).tolist() if has_cost else [0] * len(merged_metrics)
    plots["cost_latency"] = _plot_scatter(lat, cost, all_labels, "Avg Latency (ms)", "Estimated Cost", "Cost / Latency Trade-off")

    from judge_eval.metrics import compute_rankings
    global_rankings = compute_rankings(merged_metrics)
    if not global_rankings.empty:
        plots["rankings"] = _plot_model_rankings_chart(global_rankings)

    if not merged_ps.empty:
        ps_pivot = merged_ps.pivot_table(
            index="judge_model",
            columns="prompt_left",
            values="label_flip_rate",
            aggfunc="mean",
            fill_value=0.0,
        )
        plots["prompt_sensitivity_heatmap"] = _plot_heatmap(ps_pivot, "Prompt Sensitivity — Label Flip Rate", cmap="YlOrRd")

    if not merged_dr.empty:
        dr_labels = (merged_dr["judge_model"] + ":" + merged_dr["dummy_class"]).tolist()
        plots["dummy_robustness"] = _plot_bar(dr_labels, merged_dr["robustness_accuracy"].tolist(), "Dummy Answer Robustness", "Accuracy", "#7C3AED")

    # Legend for experiment colors
    legend_items = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:0.4rem;margin-right:1rem;">'
        f'<span style="display:inline-block;width:14px;height:14px;border-radius:3px;background:{color_map[n]};"></span>'
        f'{n}</span>'
        for n in sorted(color_map)
    )
    legend_html = f'<div style="margin-bottom:1rem;font-size:0.88rem;">{legend_items}</div>'

    def section(title: str, content: str) -> str:
        return f'<section class="section-block"><h2>{title}</h2>{content}</section>'

    def chart_section(title: str, plot_key: str, desc: str = "") -> str:
        img = plots.get(plot_key, "")
        img_tag = (
            f'<div class="chart-wrap"><img src="data:image/png;base64,{img}" alt="{title}" /></div>'
            if img else "<p class='no-data'>데이터 없음</p>"
        )
        desc_tag = f'<p class="metric-desc">{desc}</p>' if desc else ""
        return f'<section class="metric-card"><h2>{title}</h2>{desc_tag}{img_tag}</section>'

    display_cols = ["experiment", "judge_model", "prompt_template", "scotts_pi", "percent_agreement", "precision", "recall", "f1", "fpr", "fnr", "score_delta", "invalid_rate", "total_estimated_cost"]
    display_cols = [c for c in display_cols if c in merged_metrics.columns]
    table_df = merged_metrics[display_cols].sort_values("scotts_pi", ascending=False).round(4)

    ps_models = merged_ps["judge_model"].nunique() if not merged_ps.empty else 0
    dr_models = merged_dr["judge_model"].nunique() if not merged_dr.empty else 0

    body_parts = [
        '<h1 class="section-title">실험 목록</h1>',
        section("병합된 실험", (
            f"<p style='margin-bottom:0.8rem'>총 <strong>{len(output_dirs)}</strong>개 실험 &nbsp;·&nbsp; "
            f"<strong>{merged_metrics['judge_model'].nunique()}</strong>개 모델</p>"
            + legend_html
            + _html_table(pd.DataFrame({"실험명": exp_names, "디렉토리": [d.name for d in output_dirs]}))
        )),
        '<h1 class="section-title">핵심 성능 지표 (모델별 최고 성능 기준)</h1>',
        chart_section("Scott's π", "scotts_pi", METRIC_DESCRIPTIONS["scotts_pi"]["desc"]),
        chart_section("Percent Agreement", "percent_agreement", METRIC_DESCRIPTIONS["percent_agreement"]["desc"]),
        chart_section("F1", "f1", METRIC_DESCRIPTIONS["precision_recall_f1"]["desc"]),
        chart_section("FPR / FNR", "fpr_fnr", METRIC_DESCRIPTIONS["fpr_fnr"]["desc"]),
        chart_section("Cost / Latency Trade-off", "cost_latency", METRIC_DESCRIPTIONS["cost_latency"]["desc"]),
        '<h1 class="section-title">전체 모델 랭킹</h1>',
        section("Model Rankings (전체)", (
            f'<div class="chart-wrap"><img src="data:image/png;base64,{plots["rankings"]}" /></div>'
            if "rankings" in plots else "<p class='no-data'>데이터 없음</p>"
        )),
        f'<h1 class="section-title">프롬프트 민감도 ({ps_models}개 모델 — 복수 프롬프트 실험만 포함)</h1>',
        (
            chart_section("Prompt Sensitivity", "prompt_sensitivity_heatmap", METRIC_DESCRIPTIONS["prompt_sensitivity"]["desc"])
            + section(
                "프롬프트별 상세 비교",
                _html_table(
                    merged_ps[["experiment", "judge_model", "prompt_left", "prompt_right", "label_flip_rate", "scotts_pi_by_prompt", "prompt_consistency"]].round(4)
                ),
            )
        ) if not merged_ps.empty else section("프롬프트 민감도", "<p class='no-data'>해당 데이터 없음 — 복수 프롬프트 실험 없음</p>"),
        f'<h1 class="section-title">더미 응답 강건성 ({dr_models}개 모델 — 더미 테스트 실험만 포함)</h1>',
        (
            chart_section("Dummy Answer Robustness", "dummy_robustness", METRIC_DESCRIPTIONS["dummy_robustness"]["desc"])
            + section("더미 응답 상세", _html_table(merged_dr))
        ) if not merged_dr.empty else section("더미 응답 강건성", "<p class='no-data'>해당 데이터 없음</p>"),
        '<h1 class="section-title">전체 메트릭 테이블</h1>',
        section("All Metrics (Scott's π 내림차순)", _html_table(table_df)),
    ]

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Merged Judge Evaluation Report</title>
  <style>{_CSS}</style>
</head>
<body>
  <header>
    <h1>Merged Judge Evaluation Report</h1>
    <p>{len(output_dirs)}개 실험 병합 &nbsp;|&nbsp; {merged_metrics['judge_model'].nunique()}개 모델</p>
  </header>
  <div class="container">
    {"".join(body_parts)}
  </div>
  <footer>
    Generated by judge-model-evaluation — merged report
  </footer>
</body>
</html>
"""

    report_path.write_text(html, encoding="utf-8")
    return report_path


def generate_report(output_dir: Path) -> Path:
    metrics = _safe_read_csv(output_dir / "metrics_overall.csv")
    rankings = _safe_read_csv(output_dir / "model_rankings.csv")
    prompt_sensitivity = _safe_read_csv(output_dir / "prompt_sensitivity.csv")
    reference_order = _safe_read_csv(output_dir / "reference_order_sensitivity.csv")
    dummy_robustness = _safe_read_csv(output_dir / "dummy_answer_robustness.csv")
    answer_source_metrics = _safe_read_csv(output_dir / "metrics_by_answer_source.csv")
    metrics_by_dataset = _safe_read_csv(output_dir / "metrics_by_dataset.csv")
    leniency = _safe_read_csv(output_dir / "leniency_bias.csv")
    normalized_samples = _safe_read_parquet(output_dir / "normalized_samples.parquet")
    parsed_predictions = _safe_read_parquet(output_dir / "parsed_predictions.parquet")

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

    if _has_reference_order_visual_data(reference_order):
        plots["reference_order_visual"] = _plot_reference_order_visual(reference_order)

    paper_ref_plots = _build_paper_reference_plots()

    # ------------------------------------------------------------------
    # Collect experiment metadata
    # ------------------------------------------------------------------
    config_path = output_dir / "config.resolved.yaml"
    dataset_meta_path = output_dir / "dataset_meta.yaml"
    exp_name = output_dir.name
    dataset_meta: dict = {}
    try:
        import yaml
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        exp_name = cfg.get("experiment_name", exp_name)
        dataset_meta = yaml.safe_load(dataset_meta_path.read_text(encoding="utf-8")) if dataset_meta_path.exists() else {}
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Build callouts
    # ------------------------------------------------------------------
    overview_items = _experiment_overview_items(
        dataset_meta=dataset_meta,
        normalized_samples=normalized_samples,
        parsed_predictions=parsed_predictions,
        metrics=metrics,
        metrics_by_dataset=metrics_by_dataset,
    )
    overview_html = "".join(
        f'<div class="summary-item"><div class="label">{label}</div><div class="value">{value}</div></div>'
        for label, value in overview_items
    )

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
        '<h1 class="section-title">실험 개요</h1>',
        f'<div class="summary-grid">{overview_html}</div>',
        '<h1 class="section-title">핵심 요약</h1>',
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
            "순서 민감도 시각화",
            (
                f'<div class="chart-wrap"><img src="data:image/png;base64,{plots["reference_order_visual"]}" /></div>'
                "<p style='margin-top:0.8em;font-size:0.88em;color:#555;'>"
                "<b>순서 일관성(게이지)</b>: alias 순서를 바꿔도 동일한 레이블을 내리는 비율 — 90% 이상이면 양호. <br>"
                "<b>샘플 적용 범위(도넛)</b>: alias가 2개 이상인 샘플만 테스트 가능, 1개짜리는 제외. <br>"
                "<b>라벨 플립 현황(바)</b>: 테스트된 샘플 중 순서 변경 시 판정이 뒤집힌 건수, 점선은 10% 기준선."
                "</p>"
            )
            if "reference_order_visual" in plots
            else "<p class='no-data'>데이터 없음</p>",
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
        section(
            "Model Rankings",
            (
                f'<div class="chart-wrap"><img src="data:image/png;base64,{_plot_model_rankings_chart(rankings)}" alt="Model Rankings" /></div>'
                if not rankings.empty
                else "<p class='no-data'>데이터 없음</p>"
            ),
        ),
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
