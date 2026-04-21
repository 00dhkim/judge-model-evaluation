from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pandas as pd

from judge_eval.config import ExperimentConfig
from judge_eval.settings import ANSWER_SOURCES
from judge_eval.utils import read_json, stable_hash


def split_aliases(golden_answer: str) -> list[str]:
    return [part.strip() for part in str(golden_answer).split("/") if part.strip()]


def answer_length_bucket(candidate_answer: str) -> str:
    length = len(candidate_answer.split())
    if length < 8:
        return "short"
    if length < 25:
        return "medium"
    return "long"


def coerce_human_label(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def is_empty_candidate(value: Any) -> bool:
    if value is None:
        return True
    normalized = str(value).strip()
    return normalized == "" or normalized == "None"


def sample_id(dataset: str, row_index: int, answer_source: str) -> str:
    return stable_hash({"dataset": dataset, "row_index": row_index, "answer_source": answer_source})[:16]


def load_evouna_samples(
    config: ExperimentConfig,
    sample_size: int | None = None,
    seed: int | None = None,
    answer_sources: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    requested_sources = answer_sources or list(ANSWER_SOURCES)
    rows: list[dict[str, Any]] = []
    filter_meta = {
        "exclude_improper": not config.filter.improper,
        "exclude_non_boolean_labels": config.filter.exclude_non_boolean_labels,
        "exclude_empty_candidate_answers": config.filter.exclude_empty_candidate_answers,
    }
    for dataset_config in config.datasets:
        dataset_name = dataset_config.name.replace("evouna_", "").upper()
        dataset_rows = read_json(Path(dataset_config.path))
        for row_index, row in enumerate(dataset_rows):
            improper = bool(row.get("improper", False))
            if improper and not config.filter.improper:
                continue
            aliases = split_aliases(row["golden_answer"])
            for source in requested_sources:
                candidate = row.get(f"answer_{source}")
                human_label = coerce_human_label(row.get(f"judge_{source}"))
                if config.filter.exclude_non_boolean_labels and human_label is None:
                    continue
                if config.filter.exclude_empty_candidate_answers and is_empty_candidate(candidate):
                    continue
                rows.append(
                    {
                        "sample_id": sample_id(dataset_name, row_index, source),
                        "dataset": dataset_name,
                        "question": row["question"],
                        "golden_answer": row["golden_answer"],
                        "golden_aliases": aliases,
                        "answer_source": source,
                        "candidate_answer": str(candidate),
                        "human_label": bool(human_label),
                        "improper": improper,
                        "answer_length_bucket": answer_length_bucket(str(candidate)),
                        "golden_answer_alias_count": len(aliases),
                        "dataset_row_index": row_index,
                        "source_answer_field": f"answer_{source}",
                        "source_judge_field": f"judge_{source}",
                    }
                )
    if sample_size and sample_size < len(rows):
        rng = random.Random(seed)
        rows = rng.sample(rows, sample_size)
        rows.sort(key=lambda item: item["sample_id"])
    frame = pd.DataFrame(rows)
    dataset_hash = stable_hash({"rows": frame.to_dict(orient="records"), "filters": filter_meta})
    meta = {
        "dataset_hash": dataset_hash,
        "row_count": len(frame),
        "filter_policy": filter_meta,
        "answer_sources": requested_sources,
    }
    return frame, meta
