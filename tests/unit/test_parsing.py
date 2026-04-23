from judge_eval.parsing import parse_model_output


def test_parse_json_output():
    parsed = parse_model_output('{"reason":"ok","label":true}')
    assert parsed["parse_method"] == "json"
    assert parsed["parsed_label"] is True
    assert parsed["judge_reason"] == "ok"


def test_parse_json_output_after_thought_block():
    parsed = parse_model_output(
        '<thought>internal reasoning</thought>```json\n{"reason":"ok","label":true}\n```'
    )
    assert parsed["parse_method"] == "json"
    assert parsed["parsed_label"] is True
    assert parsed["judge_reason"] == "ok"


def test_parse_json_output_after_thought_block_without_fence():
    parsed = parse_model_output(
        '<thought>internal reasoning</thought>{"reason":"still ok","label":false}'
    )
    assert parsed["parse_method"] == "json"
    assert parsed["parsed_label"] is False
    assert parsed["judge_reason"] == "still ok"


def test_parse_regex_output():
    parsed = parse_model_output("reason\nlabel: false")
    assert parsed["parse_method"] == "regex"
    assert parsed["parsed_label"] is False


def test_parse_invalid_output():
    parsed = parse_model_output("nonsense")
    assert parsed["parse_method"] == "invalid"
    assert parsed["parsed_label"] is None


def test_parse_non_object_json_output_is_invalid():
    parsed = parse_model_output('["unexpected"]')
    assert parsed["parse_method"] == "invalid"
    assert parsed["parsed_label"] is None
