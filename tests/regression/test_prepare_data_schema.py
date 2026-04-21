from pathlib import Path

import pandas as pd

from judge_eval.config import load_config
from judge_eval.data import load_evouna_samples


def test_prepare_data_schema_snapshot():
    config, _ = load_config(Path("configs/examples/full_evouna.yaml"))
    frame, meta = load_evouna_samples(config, sample_size=5, seed=42)
    assert list(frame.columns) == [
        "sample_id",
        "dataset",
        "question",
        "golden_answer",
        "golden_aliases",
        "answer_source",
        "candidate_answer",
        "human_label",
        "improper",
        "answer_length_bucket",
        "golden_answer_alias_count",
        "dataset_row_index",
        "source_answer_field",
        "source_judge_field",
    ]
    assert meta["row_count"] == 5
    assert pd.Series(frame["sample_id"]).is_unique
