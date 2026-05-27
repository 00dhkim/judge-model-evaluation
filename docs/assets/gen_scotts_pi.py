"""README용 Scott's π 차트 생성 스크립트. 프로젝트 루트에서 실행."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import base64
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from judge_eval.reporting import _model_color, _fig_to_base64

DIRS = [
    "outputs/20260422_gemma_hosted_4way_free",
    "outputs/20260423_cerebras_hosted_2way_free",
    "outputs/20260424_openai_small_models_4way",
    "outputs/20260509_frontier_latest_202605",
]
EXCLUDE = {"kimi_k2_6", "mimo_v2_5_pro", "k_exaone_236b_a23b"}
SUBTITLE = "우연 일치를 보정한 model-human 일치도"
OPTIMIZED_REPORT = Path("outputs/merged_report_optimized.html")


def set_korean_font() -> None:
    for font_path in [
        Path.home() / ".local/share/fonts/malgun.ttf",
        Path.home() / ".local/share/fonts/malgunbd.ttf",
    ]:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
            return


def build_chart(
    dirs: list[str],
    *,
    exclude: set[str],
    out: Path,
    title: str,
    subtitle: str,
) -> None:
    dfs = []
    for d in dirs:
        p = Path(d) / "metrics_overall.csv"
        if p.exists():
            dfs.append(pd.read_csv(p))

    if not dfs:
        raise RuntimeError("no metrics_overall.csv files found")

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged[~merged["judge_model"].isin(exclude)]

    best = (
        merged.sort_values("scotts_pi", ascending=False)
        .drop_duplicates("judge_model")
        .sort_values("scotts_pi", ascending=True)
        .reset_index(drop=True)
    )

    labels = best["judge_model"].tolist()
    values = best["scotts_pi"].tolist()
    colors = [_model_color(m) for m in labels]

    fig, ax = plt.subplots(figsize=(10, 10))
    bars = ax.barh(labels, values, color=colors, height=0.7)

    for bar, val in zip(bars, values):
        ax.text(
            val + 0.008,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=10,
            color="#333333",
        )

    ax.axvline(0.8, color="#888", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Scott's π", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=26)
    ax.text(
        0.5,
        1.01,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
        color="#555555",
    )
    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out.write_bytes(base64.b64decode(_fig_to_base64(fig)))
    plt.close(fig)
    print(f"saved -> {out}")


def extract_optimized_chart(out: Path) -> None:
    html = OPTIMIZED_REPORT.read_text(encoding="utf-8")
    match = re.search(
        r"<h2>Scott's π</h2>.*?<img src=\"data:image/png;base64,([^\"]+)\"",
        html,
        flags=re.S,
    )
    if not match:
        raise RuntimeError(f"Scott's π chart not found in {OPTIMIZED_REPORT}")

    out.write_bytes(base64.b64decode(match.group(1)))
    print(f"saved -> {out}")


set_korean_font()
build_chart(
    DIRS,
    exclude=EXCLUDE,
    out=Path(__file__).parent / "scotts_pi.png",
    title="Scott's π",
    subtitle=SUBTITLE,
)
extract_optimized_chart(Path(__file__).parent / "scotts_pi_optimized.png")
