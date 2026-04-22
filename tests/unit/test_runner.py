from pathlib import Path

from judge_eval.runner import finalized_keys, finalized_keys_from_raw_jsonl


def test_finalized_keys_from_raw_jsonl_reads_unit_keys(tmp_path: Path):
    raw_path = tmp_path / "raw_predictions.jsonl"
    raw_path.write_text(
        "\n".join(
            [
                '{"unit_key":"u1","parse_status":"ok"}',
                "",
                '{"unit_key":"u2","parse_status":"error"}',
                '{"parse_status":"ok"}',
                "{bad json",
            ]
        ),
        encoding="utf-8",
    )

    assert finalized_keys_from_raw_jsonl(raw_path) == {"u1", "u2"}


def test_finalized_keys_falls_back_to_raw_when_parquet_missing(tmp_path: Path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "raw_predictions.jsonl").write_text(
        '{"unit_key":"resume-me","parse_status":"ok"}\n',
        encoding="utf-8",
    )

    assert finalized_keys(output_dir) == {"resume-me"}
