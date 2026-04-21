from judge_eval.parsing import parse_model_output


def test_parse_json_output():
    parsed = parse_model_output('{"reason":"ok","label":true}')
    assert parsed["parse_method"] == "json"
    assert parsed["parsed_label"] is True


def test_parse_regex_output():
    parsed = parse_model_output("reason\nlabel: false")
    assert parsed["parse_method"] == "regex"
    assert parsed["parsed_label"] is False


def test_parse_invalid_output():
    parsed = parse_model_output("nonsense")
    assert parsed["parse_method"] == "invalid"
    assert parsed["parsed_label"] is None
