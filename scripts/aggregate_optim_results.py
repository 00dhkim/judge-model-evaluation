"""Aggregate optimization predictions into a full leaderboard.

This script treats per-method predictions.jsonl files as the source of truth.
It scans every immediate child directory under --root, merges predictions with
the matching normalized_samples.parquet, and writes:

  - summary_full.csv
  - summary_manifest.json
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
from typing import Any

import pandas as pd

from judge_eval.metrics import compute_metric_result
from judge_eval.utils import stable_hash, write_json


DEFAULT_SAMPLE_SOURCES = {
    "solar": Path("outputs/20260523_solar_202605_tuned/normalized_samples.parquet"),
    "exaone": Path("outputs/20260523_exaone_202605_tuned/normalized_samples.parquet"),
}


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def read_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_json(path, lines=True)
    if frame.empty:
        return frame
    if "sample_id" not in frame.columns:
        raise ValueError(f"{path} is missing required column: sample_id")
    if "parsed_label" not in frame.columns:
        raise ValueError(f"{path} is missing required column: parsed_label")
    return frame


def load_sample_labels(sample_source: Path) -> pd.DataFrame:
    samples = pd.read_parquet(sample_source)
    required = {"sample_id", "human_label"}
    missing = required - set(samples.columns)
    if missing:
        raise ValueError(f"{sample_source} is missing required column(s): {sorted(missing)}")
    labels = samples[["sample_id", "human_label"]].copy()
    if labels["sample_id"].duplicated().any():
        duplicates = labels.loc[labels["sample_id"].duplicated(), "sample_id"].head(5).tolist()
        raise ValueError(f"{sample_source} has duplicate sample_id values, e.g. {duplicates}")
    return labels


def sample_hash(labels: pd.DataFrame) -> str:
    normalized = labels.sort_values("sample_id")[["sample_id", "human_label"]].to_dict(orient="records")
    return stable_hash(normalized)


def _cost_series(frame: pd.DataFrame) -> pd.Series:
    if "total_cost" in frame.columns:
        return frame["total_cost"]
    if "cost" in frame.columns:
        return frame["cost"]
    if "estimated_cost" in frame.columns:
        return frame["estimated_cost"]
    return pd.Series([0.0] * len(frame), index=frame.index)


def _latency_series(frame: pd.DataFrame) -> pd.Series:
    if "total_latency_ms" in frame.columns:
        return frame["total_latency_ms"]
    if "latency_ms" in frame.columns:
        return frame["latency_ms"]
    return pd.Series([0.0] * len(frame), index=frame.index)


def aggregate_prediction_file(
    predictions_path: Path,
    labels: pd.DataFrame,
    *,
    judge: str,
    mode: str,
    sample_source: Path,
    sample_hash_value: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    predictions = read_predictions(predictions_path)
    method = predictions_path.parent.name
    merged = predictions.merge(labels, on="sample_id", how="left", validate="many_to_one")
    missing_labels = int(merged["human_label"].isna().sum())
    if missing_labels:
        raise ValueError(f"{predictions_path} has {missing_labels} prediction row(s) with no matching sample label")

    metric_frame = pd.DataFrame(
        {
            "parsed_label": merged["parsed_label"],
            "human_label": merged["human_label"],
            "latency_ms": _latency_series(merged),
            "estimated_cost": _cost_series(merged),
        }
    )
    metrics = compute_metric_result(metric_frame)
    valid_count = int(metric_frame["parsed_label"].notna().sum())
    total_calls = int(merged["n_calls"].sum()) if "n_calls" in merged.columns else 0
    total_cost = float(_cost_series(merged).fillna(0.0).sum())
    summary_row = {
        "judge": judge,
        "mode": mode,
        "method": method,
        "sample_count": int(len(merged)),
        "unique_sample_count": int(merged["sample_id"].nunique()) if "sample_id" in merged.columns else 0,
        "valid_count": valid_count,
        "coverage": metrics.valid_rate,
        "scotts_pi": metrics.scotts_pi,
        "accuracy": metrics.percent_agreement,
        "total_calls": total_calls,
        "total_cost": total_cost,
        "avg_latency_ms": metrics.avg_latency_ms,
        "predictions_path": str(predictions_path),
        "sample_source_path": str(sample_source),
        "sample_hash": sample_hash_value,
    }
    manifest_entry = {
        "method": method,
        "predictions_path": str(predictions_path),
        "row_count": int(len(predictions)),
        "unique_sample_count": int(predictions["sample_id"].nunique()) if "sample_id" in predictions.columns else 0,
        "valid_count": valid_count,
        "missing_label_count": missing_labels,
    }
    return summary_row, manifest_entry


def aggregate_root(
    root: Path,
    *,
    judge: str,
    sample_source: Path,
    output_csv: Path | None = None,
    output_manifest: Path | None = None,
    generated_at: str | None = None,
    commit: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    labels = load_sample_labels(sample_source)
    hash_value = sample_hash(labels)
    mode = root.name
    generated_at_value = generated_at or datetime.now(UTC).isoformat(timespec="seconds")
    commit_value = commit or git_commit()
    prediction_paths = sorted(root.glob("*/predictions.jsonl"))
    if not prediction_paths:
        raise ValueError(f"no predictions.jsonl files found under {root}")

    rows: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    for predictions_path in prediction_paths:
        row, entry = aggregate_prediction_file(
            predictions_path,
            labels,
            judge=judge,
            mode=mode,
            sample_source=sample_source,
            sample_hash_value=hash_value,
        )
        row["generated_at"] = generated_at_value
        row["git_commit"] = commit_value
        rows.append(row)
        files.append(entry)

    summary = pd.DataFrame(rows).sort_values(["scotts_pi", "method"], ascending=[False, True]).reset_index(drop=True)
    manifest = {
        "root": str(root),
        "judge": judge,
        "mode": mode,
        "sample_source_path": str(sample_source),
        "sample_hash": hash_value,
        "generated_at": generated_at_value,
        "git_commit": commit_value,
        "summary_path": str(output_csv or root / "summary_full.csv"),
        "files": files,
    }

    csv_path = output_csv or root / "summary_full.csv"
    manifest_path = output_manifest or root / "summary_manifest.json"
    summary.to_csv(csv_path, index=False)
    write_json(manifest_path, manifest)
    return summary, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate optimization predictions into summary_full.csv")
    parser.add_argument("--root", required=True, type=Path, help="Optimization output root containing */predictions.jsonl")
    parser.add_argument("--judge", required=True, choices=sorted(DEFAULT_SAMPLE_SOURCES), help="Judge sample source preset")
    parser.add_argument("--sample-source", type=Path, help="Override normalized_samples.parquet path")
    parser.add_argument("--output-csv", type=Path, help="Output CSV path (default: ROOT/summary_full.csv)")
    parser.add_argument("--output-manifest", type=Path, help="Output manifest path (default: ROOT/summary_manifest.json)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_source = args.sample_source or DEFAULT_SAMPLE_SOURCES[args.judge]
    summary, manifest = aggregate_root(
        args.root,
        judge=args.judge,
        sample_source=sample_source,
        output_csv=args.output_csv,
        output_manifest=args.output_manifest,
    )
    print(f"Wrote {manifest['summary_path']} ({len(summary)} method(s))")


if __name__ == "__main__":
    main()
