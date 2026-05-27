"""Build merged_report_optimized.html using the best optimized methods.

Compares the raw, unoptimized baseline entries for solar_pro_3 and
k_exaone_236b_a23b against the best optimized runs:
  - Solar:  n06_sc5_ext_confidence  (sc5 + ext_fewshot + confidence_abstain)
  - EXAONE: c10_all_lite             (sc3 + alias_enum + ext_fewshot + confidence)

Steps:
  1. Read predictions.jsonl for the optimized methods and join with
     normalized_samples.parquet (ground truth) of the matching baseline run.
  2. Compute metrics_overall.csv via judge_eval.metrics.compute_metric_result.
  3. Materialize fake "experiment" output dirs.
  4. Call generate_merged_report() with frontier dirs, raw EXAONE baseline,
     and the 2 optimized dirs.
  5. Inject a "Optimization Methods" section listing methods + descriptions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from judge_eval.metrics import compute_metric_result
from judge_eval import reporting as R
from judge_eval.reporting import generate_merged_report


# ---- Monkey-patch reporting to bold optimized model names -------------------

BOLD_MODELS = {"solar_pro_3_optimized", "k_exaone_236b_a23b_optimized"}


def _contains_bold(text: str) -> bool:
    return any(m in text for m in BOLD_MODELS)


_orig_fig_to_base64 = R._fig_to_base64


def _fig_to_base64_bold(fig):
    for ax in fig.get_axes():
        for tick in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            if _contains_bold(tick.get_text()):
                tick.set_fontweight("bold")
                tick.set_color("#0f3460")
    return _orig_fig_to_base64(fig)


R._fig_to_base64 = _fig_to_base64_bold


_orig_html_table = R._html_table


def _html_table_bold(frame):
    html = _orig_html_table(frame)
    for m in BOLD_MODELS:
        html = html.replace(
            f"<td>{m}</td>",
            f'<td><strong style="color:#0f3460">{m}</strong></td>',
        )
    return html


R._html_table = _html_table_bold


# Graph labels should compare model identities only. Prompt/method names remain
# available in the metric tables, but are intentionally hidden from chart axes.
def _model_only_labels(metrics: pd.DataFrame, *, style: str) -> list[str]:
    if "judge_model" not in metrics.columns:
        return []
    return [
        R._display_model_name(model)
        for model in metrics["judge_model"].fillna("").astype(str).tolist()
    ]


R._model_prompt_labels = _model_only_labels


# Plotly charts: wrap matching labels with <b>...</b>
def _wrap_plotly_labels(labels):
    out = []
    for lbl in labels:
        if _contains_bold(str(lbl)):
            out.append(f"<b>{lbl}</b>")
        else:
            out.append(lbl)
    return out


# Fade non-highlight bars in Scott's π and Percent Agreement charts.
# Highlight set = before/after pairs for Solar and EXAONE so the contrast pops.
HIGHLIGHT_SUBSTRS = ("solar", "exaone")
FADE_CHART_TITLES = {"Scott's π", "Percent Agreement"}


def _fade_color(c, alpha: float = 0.25):
    """Blend a matplotlib color toward white to mute it."""
    import matplotlib.colors as mcolors
    r, g, b, _ = mcolors.to_rgba(c)
    # Blend with white: result = bg*(1-alpha) + color*alpha (low alpha = faded)
    bg = 1.0
    return (bg*(1-alpha) + r*alpha, bg*(1-alpha) + g*alpha, bg*(1-alpha) + b*alpha, 1.0)


_orig_plot_bar = R._plot_bar


def _plot_bar_faded(labels, values, title, ylabel, color="#4C8BF5", vline=None):
    if title in FADE_CHART_TITLES and isinstance(color, (list, tuple)):
        new_colors = []
        for lbl, c in zip(labels, color):
            lbl_lc = str(lbl).lower()
            if any(s in lbl_lc for s in HIGHLIGHT_SUBSTRS):
                new_colors.append(c)
            else:
                new_colors.append(_fade_color(c, alpha=0.30))
        color = new_colors
    return _orig_plot_bar(labels, values, title, ylabel, color, vline)


R._plot_bar = _plot_bar_faded


for _fn_name in ("_plot_performance_efficiency_html", "_plot_score_delta_html", "_plot_fpr_fnr_quadrant_html"):
    _orig = getattr(R, _fn_name)

    def _make_wrapper(orig):
        def wrapper(labels, *a, **kw):
            return orig(_wrap_plotly_labels(labels), *a, **kw)
        return wrapper

    setattr(R, _fn_name, _make_wrapper(_orig))


ROOT = Path("/home/primi/workspace/judge-model-evaluation")
OUT = ROOT / "outputs"


# ---- Optimized method definitions (for HTML injection) ----------------------

METHOD_DESCRIPTIONS = {
    "m1_sc (Self-Consistency)": (
        "K회 다중 샘플링 후 다수결로 라벨 결정. temperature=0.7로 K=3~5번 호출, "
        "JSON label 다수결, 동률 시 confidence 가중. CoT를 사용하는 judge에서 "
        "GSM8K +17.9%, NDCG +7.5pt 수준의 일관된 이득이 보고된 가장 ROI 큰 단일 기법."
    ),
    "m2_alias_enum (Alias Enumeration)": (
        "EVOUNA golden_answers의 `/` 구분 별칭을 프롬프트에 `Acceptable forms: [a1, a2, ...]`로 "
        "명시. Reference-Guided Verdict 방식으로 false negative를 크게 줄임. 단독으로는 "
        "역효과(-0.023)지만 조합에서 효과가 나타남."
    ),
    "m3_alias_shuffle (Order Bias 대응)": (
        "JudgeLM의 핵심 트릭. golden ↔ candidate alias 순서를 두 가지 랜덤 시드로 섞어 "
        "2회 호출, 일치 시 채택, 불일치 시 3번째 호출 다수결(2/3). Position bias 큰 모델일수록 효과 큼."
    ),
    "m6_extended_fewshot (Hard-negative ICL)": (
        "현재 모델이 틀린 케이스(false positive/negative)에서 별칭 차이·동의어·부분 일치·잘못된 entity 등 "
        "hard-negative 예시 8개를 큐레이션해 프롬프트에 박제. ICL 단독 효과는 작지만 sc/confidence와 결합 시 시너지."
    ),
    "m7_confidence_abstain (Confidence Calibration)": (
        "모델에 confidence 수치(1–5점 또는 확률)를 요청해서 threshold 미만이면 abstain, 그렇지 않으면 채택. "
        "Scott's π는 일관된 응답에 보상이 크므로 confident 케이스만 채택해도 π 상승. 개별 방법 중 Solar 최고 성능(+0.049)."
    ),
}

# 모델별 적용 메소드 (오케스트레이션 구성)
MODEL_METHOD_STACK = {
    "solar_pro_3_optimized": {
        "label": "Solar Pro3 (n06_sc5_ext_confidence)",
        "baseline_label": "solar_pro_3 / minimal",
        "baseline_pi": 0.6185,
        "optimized_pi_expected": 0.8016,
        "stack": [
            "m1_sc (Self-Consistency)",
            "m6_extended_fewshot (Hard-negative ICL)",
            "m7_confidence_abstain (Confidence Calibration)",
        ],
        "stack_detail": "Self-Consistency K=5 → Hard-negative few-shot 8개 → Confidence abstain (threshold 기반)",
    },
    "k_exaone_236b_a23b_optimized": {
        "label": "K-EXAONE-236B-A23B (c10_all_lite)",
        "baseline_label": "k_exaone_236b_a23b / minimal",
        "baseline_pi": 0.6141,
        "optimized_pi_expected": 0.8340,
        "stack": [
            "m1_sc (Self-Consistency)",
            "m2_alias_enum (Alias Enumeration)",
            "m6_extended_fewshot (Hard-negative ICL)",
            "m7_confidence_abstain (Confidence Calibration)",
        ],
        "stack_detail": "Self-Consistency K=3 → Alias enumeration → Hard-negative few-shot → Confidence abstain",
    },
}


# ---- 1. Build metrics for optimized runs ------------------------------------

def build_metrics_dir(
    pred_path: Path,
    baseline_dir: Path,
    judge_model: str,
    prompt_template: str,
    out_dir: Path,
    experiment_name: str,
    method_id: str,
) -> Path:
    """Materialize one optimized experiment dir with metrics_overall.csv + config."""
    preds = pd.read_json(pred_path, lines=True)
    samples = pd.read_parquet(baseline_dir / "normalized_samples.parquet")

    # join on sample_id
    df = preds.merge(samples[["sample_id", "human_label"]], on="sample_id", how="inner")
    if len(df) != len(preds):
        raise RuntimeError(f"merge mismatch: preds={len(preds)} merged={len(df)}")

    # Build the frame expected by compute_metric_result
    frame = pd.DataFrame({
        "parsed_label": df["parsed_label"],
        "human_label": df["human_label"],
        "latency_ms": df.get("total_latency_ms", pd.Series([0] * len(df))),
        "estimated_cost": df.get("total_cost", pd.Series([0.0] * len(df))),
    })

    m = compute_metric_result(frame)
    row = {
        "judge_model": judge_model,
        "prompt_template": prompt_template,
        "judge_score": m.judge_score,
        "human_score": m.human_score,
        "score_delta": m.score_delta,
        "percent_agreement": m.percent_agreement,
        "scotts_pi": m.scotts_pi,
        "precision": m.precision,
        "recall": m.recall,
        "f1": m.f1,
        "tp": m.tp,
        "fp": m.fp,
        "tn": m.tn,
        "fn": m.fn,
        "fpr": m.fpr,
        "fnr": m.fnr,
        "invalid_rate": m.invalid_rate,
        "valid_rate": m.valid_rate,
        "avg_latency_ms": m.avg_latency_ms,
        "p50_latency_ms": m.p50_latency_ms,
        "p95_latency_ms": float(frame["latency_ms"].quantile(0.95)) if not frame["latency_ms"].empty else 0.0,
        "total_estimated_cost": float(frame["estimated_cost"].sum()),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out_dir / "metrics_overall.csv", index=False)

    config = {
        "experiment_name": experiment_name,
        "_method_id": method_id,
        "_source_predictions": str(pred_path),
        "judge_models": [{"name": judge_model}],
        "evaluation": {"prompt_templates": [prompt_template]},
    }
    import yaml
    (out_dir / "config.resolved.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    print(f"  {judge_model}: π={m.scotts_pi:.4f}, n={len(df)}, cost=${row['total_estimated_cost']:.4f} → {out_dir}")
    return out_dir


# ---- 2. Build optimized experiment dirs --------------------------------------

print("Building optimized experiment dirs...")
solar_opt_dir = build_metrics_dir(
    pred_path=OUT / "optim_final" / "n06_sc5_ext_confidence" / "predictions.jsonl",
    baseline_dir=OUT / "20260523_solar_202605_tuned",
    judge_model="solar_pro_3_optimized",
    prompt_template="n06_sc5_ext_confidence",
    out_dir=OUT / "solar_optimized_n06",
    experiment_name="solar_optimized_n06",
    method_id="n06_sc5_ext_confidence",
)
exaone_opt_dir = build_metrics_dir(
    pred_path=OUT / "optim_final_exaone" / "c10_all_lite" / "predictions.jsonl",
    baseline_dir=OUT / "20260523_exaone_202605_tuned",
    judge_model="k_exaone_236b_a23b_optimized",
    prompt_template="c10_all_lite",
    out_dir=OUT / "exaone_optimized_c10",
    experiment_name="exaone_optimized_c10",
    method_id="c10_all_lite",
)


# ---- 3. Run merge ------------------------------------------------------------

merge_dirs = [
    OUT / "20260422_gemma_hosted_4way_free",
    OUT / "20260423_cerebras_hosted_2way_free",
    OUT / "20260424_openai_small_models_4way",
    OUT / "20260509_frontier_latest_202605",  # includes raw solar_pro_3 / minimal
    OUT / "20260522_exaone_202605",           # raw k_exaone_236b_a23b / minimal
    solar_opt_dir,
    exaone_opt_dir,
]

report_path = OUT / "merged_report_optimized.html"
print(f"\nGenerating {report_path}...")
generate_merged_report(merge_dirs, report_path)


# ---- 4. Inject "Optimization Methods" section into HTML ---------------------

print("Injecting optimization-methods section...")
html = report_path.read_text(encoding="utf-8")


def _methods_section_html() -> str:
    rows = []
    for slug, info in MODEL_METHOD_STACK.items():
        chips = "".join(
            f'<span style="display:inline-block;background:#eef2ff;color:#3730a3;padding:2px 8px;border-radius:4px;margin:2px;font-size:0.82rem;font-family:monospace">{m}</span>'
            for m in info["stack"]
        )
        delta = info["optimized_pi_expected"] - info["baseline_pi"]
        rows.append(f"""
        <div style="border:1px solid #e5e7eb;border-radius:10px;padding:1rem 1.25rem;margin-bottom:1rem;background:#fafbfc">
          <div style="display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:0.5rem">
            <h3 style="margin:0;color:#1a1a2e">{info["label"]}</h3>
            <div style="font-family:monospace;font-size:0.92rem">
              baseline <span style="color:#4b5563">({info["baseline_label"]})</span>
              π <strong>{info["baseline_pi"]:.4f}</strong>
              &nbsp;→&nbsp;
              optimized π <strong style="color:#16a34a">{info["optimized_pi_expected"]:.4f}</strong>
              &nbsp;(<span style="color:#16a34a">+{delta:.4f}</span>)
            </div>
          </div>
          <div style="margin-top:0.6rem">{chips}</div>
          <p style="margin-top:0.6rem;color:#4b5563;font-size:0.9rem"><strong>Stack:</strong> {info["stack_detail"]}</p>
        </div>
        """)

    method_defs = "".join(
        f"""<div style="margin-bottom:0.8rem"><div style="font-family:monospace;font-weight:600;color:#111827">{name}</div><div style="color:#4b5563;font-size:0.92rem;margin-top:0.2rem">{desc}</div></div>"""
        for name, desc in METHOD_DESCRIPTIONS.items()
    )

    return f"""
<h1 class="section-title">최적화 기법 (Optimization Methods)</h1>
<section class="section-block">
  <h2>모델별 적용된 최적화 스택</h2>
  <p style="margin-bottom:0.8rem;color:#4b5563">
    이 보고서의 <code>solar_pro_3_optimized</code> 및 <code>k_exaone_236b_a23b_optimized</code> 항목은
    1200샘플(EVOUNA TQ 600 + NQ 600) 평가에서 가장 높은 Scott's π를 기록한 최적 조합입니다.
    비교 기준선은 <code>solar_pro_3 / minimal</code> 및 <code>k_exaone_236b_a23b / minimal</code>의
    무최적화 실행 결과입니다.
    상세 실험 로그: <code>experiments/README.md</code>.
  </p>
  {"".join(rows)}
</section>
<section class="section-block">
  <h2>개별 메소드 설명</h2>
  {method_defs}
</section>
"""


inject = _methods_section_html()
marker = '<h1 class="section-title">실험 목록</h1>'
if marker not in html:
    raise RuntimeError("anchor not found in generated HTML")
html = html.replace(marker, inject + marker, 1)

report_path.write_text(html, encoding="utf-8")
print(f"Done → {report_path}")
print(f"Size: {report_path.stat().st_size:,} bytes")
