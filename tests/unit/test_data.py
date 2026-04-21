from judge_eval.data import answer_length_bucket, coerce_human_label, is_empty_candidate, split_aliases


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
