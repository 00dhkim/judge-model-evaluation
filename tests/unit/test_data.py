import pytest

from judge_eval.config import ExperimentConfig
from judge_eval.data import answer_length_bucket, coerce_human_label, is_empty_candidate, load_evouna_samples, split_aliases


def test_split_aliases():
    assert split_aliases("a / b/c") == ["a", "b", "c"]


def test_answer_length_bucket():
    assert answer_length_bucket("one two") == "short"
    assert answer_length_bucket(" ".join(["x"] * 10)) == "medium"
    assert answer_length_bucket(" ".join(["x"] * 30)) == "long"


def test_coerce_human_label():
    assert coerce_human_label(True) is True
    assert coerce_human_label("nan") is None


def test_is_empty_candidate():
    assert is_empty_candidate(None) is True
    assert is_empty_candidate("None") is True
    assert is_empty_candidate(" answer ") is False


def test_load_evouna_samples_warns_when_non_boolean_labels_are_kept(tmp_path):
    dataset_path = tmp_path / "sample.json"
    dataset_path.write_text(
        '[{"question":"q","golden_answer":"a","answer_model":"x","judge_model":null}]',
        encoding="utf-8",
    )
    config = ExperimentConfig.model_validate(
        {
            "experiment_name": "warn_case",
            "datasets": [{"name": "evouna_tq", "path": str(dataset_path)}],
            "filter": {
                "exclude_non_boolean_labels": False,
                "exclude_empty_candidate_answers": False,
            },
            "judge_models": [{"name": "dummy", "provider": "dummy"}],
            "output": {"experiment_name": "out"},
        }
    )
    with pytest.warns(UserWarning, match="exclude_non_boolean_labels=false"):
        frame, _ = load_evouna_samples(config, answer_sources=["model"])
    assert bool(frame.iloc[0]["human_label"]) is False
