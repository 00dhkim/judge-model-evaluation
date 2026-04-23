from pathlib import Path

import pandas as pd

from judge_eval.runner import (
    final_predictions_from_raw_jsonl,
    failed_unit_keys_from_raw_jsonl,
    finalized_keys,
    finalized_keys_from_raw_jsonl,
    load_raw_predictions,
    prepare_predictions_for_parquet,
)


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


def test_finalized_keys_prefers_raw_when_parquet_is_stale(tmp_path: Path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "raw_predictions.jsonl").write_text(
        "\n".join(
            [
                '{"unit_key":"u1","parse_status":"ok"}',
                '{"unit_key":"u2","parse_status":"error"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pd.DataFrame([{"unit_key": "u1"}]).to_parquet(output_dir / "parsed_predictions.parquet", index=False)

    assert finalized_keys(output_dir) == {"u1", "u2"}


def test_load_raw_predictions_reads_jsonl_rows(tmp_path: Path):
    raw_path = tmp_path / "raw_predictions.jsonl"
    raw_path.write_text(
        "\n".join(
            [
                '{"unit_key":"u1","variant_metadata":{}}',
                '{"unit_key":"u2","variant_metadata":{"dummy_class":"yes"}}',
            ]
        ),
        encoding="utf-8",
    )

    frame = load_raw_predictions(raw_path)

    assert frame["unit_key"].tolist() == ["u1", "u2"]
    assert frame["variant_metadata"].tolist() == [{}, {"dummy_class": "yes"}]


def test_prepare_predictions_for_parquet_normalizes_empty_variant_metadata():
    frame = pd.DataFrame(
        [
            {"unit_key": "u1", "variant_metadata": {}},
            {"unit_key": "u2", "variant_metadata": {"dummy_class": "yes"}},
            {"unit_key": "u3", "variant_metadata": None},
        ]
    )

    normalized = prepare_predictions_for_parquet(frame)

    assert normalized["variant_metadata"].tolist() == [
        None,
        {"dummy_class": "yes"},
        None,
    ]


def test_prepare_predictions_for_parquet_can_write_all_empty_variant_metadata(tmp_path: Path):
    frame = pd.DataFrame(
        [
            {"unit_key": "u1", "variant_metadata": {}},
            {"unit_key": "u2", "variant_metadata": {}},
        ]
    )

    normalized = prepare_predictions_for_parquet(frame)
    path = tmp_path / "parsed_predictions.parquet"
    normalized.to_parquet(path, index=False)
    loaded = pd.read_parquet(path)

    assert loaded["unit_key"].tolist() == ["u1", "u2"]
    assert loaded["variant_metadata"].isna().tolist() == [True, True]


def test_final_predictions_from_raw_jsonl_keeps_last_attempt_per_unit(tmp_path: Path):
    raw_path = tmp_path / "raw_predictions.jsonl"
    raw_path.write_text(
        "\n".join(
            [
                '{"unit_key":"u1","retry_count":0,"parse_status":"invalid"}',
                '{"unit_key":"u1","retry_count":1,"parse_status":"retry_ok"}',
                '{"unit_key":"u2","retry_count":0,"parse_status":"ok"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    frame = final_predictions_from_raw_jsonl(raw_path)

    assert frame["unit_key"].tolist() == ["u1", "u2"]
    assert frame["parse_status"].tolist() == ["retry_ok", "ok"]


def test_failed_unit_keys_from_raw_jsonl_uses_final_status_per_unit(tmp_path: Path):
    raw_path = tmp_path / "raw_predictions.jsonl"
    raw_path.write_text(
        "\n".join(
            [
                '{"unit_key":"u1","retry_count":0,"parse_status":"error"}',
                '{"unit_key":"u1","retry_count":1,"parse_status":"ok"}',
                '{"unit_key":"u2","retry_count":0,"parse_status":"invalid"}',
                '{"unit_key":"u3","retry_count":0,"parse_status":"error"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert failed_unit_keys_from_raw_jsonl(raw_path) == {"u2", "u3"}
