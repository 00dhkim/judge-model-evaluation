import pandas as pd

from judge_eval.cli import _exclude_models


def test_exclude_models_filters_judge_model_rows():
    frame = pd.DataFrame(
        [
            {"judge_model": "judge-a", "value": 1},
            {"judge_model": "judge-b", "value": 2},
            {"judge_model": "judge-a", "value": 3},
        ]
    )

    filtered = _exclude_models(frame, {"judge-b"})

    assert filtered["judge_model"].tolist() == ["judge-a", "judge-a"]
    assert filtered["value"].tolist() == [1, 3]


def test_exclude_models_noops_when_column_missing():
    frame = pd.DataFrame([{"value": 1}])

    assert _exclude_models(frame, {"judge-b"}).equals(frame)
