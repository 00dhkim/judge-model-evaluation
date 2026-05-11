"""docs/assets/scotts_pi.png 생성 스크립트. 프로젝트 루트에서 실행."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import base64
import pandas as pd
import matplotlib.pyplot as plt
from judge_eval.reporting import _model_color, _fig_to_base64

DIRS = [
    "outputs/20260422_gemma_hosted_4way_free",
    "outputs/20260423_cerebras_hosted_2way_free",
    "outputs/20260424_openai_small_models_4way",
    "outputs/20260509_frontier_latest_202605",
]
EXCLUDE = {"kimi_k2_6", "mimo_v2_5_pro", "k_exaone_236b_a23b"}
OUT = Path(__file__).parent / "scotts_pi.png"

dfs = []
for d in DIRS:
    p = Path(d) / "metrics_overall.csv"
    if p.exists():
        dfs.append(pd.read_csv(p))

merged = pd.concat(dfs, ignore_index=True)
merged = merged[~merged["judge_model"].isin(EXCLUDE)]

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
    ax.text(val + 0.008, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=10, color="#333333")

ax.axvline(0.8, color="#888", linewidth=1, linestyle="--", alpha=0.5)
ax.set_xlim(0, 1.02)
ax.set_xlabel("Scott's π", fontsize=12)
ax.set_title("Scott's π", fontsize=14, fontweight="bold", pad=12)
ax.tick_params(axis="y", labelsize=10)
ax.tick_params(axis="x", labelsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
OUT.write_bytes(base64.b64decode(_fig_to_base64(fig)))
print(f"saved → {OUT}")
