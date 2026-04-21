from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

from judge_eval.config import (
    config_hash,
    load_config,
    resolve_output_dir,
    resolved_config_with_redactions,
    validate_config_file,
    write_output_dir_manifest,
)
from judge_eval.data import load_evouna_samples
from judge_eval.metrics import write_metrics_bundle
from judge_eval.reporting import generate_report
from judge_eval.runner import prepare_output_dir, run_predictions, write_resolved_config
from judge_eval.telemetry import (
    build_metrics_dataset_file,
    load_telemetry_manifest,
    sync_ax_dataset,
    write_telemetry_manifest,
)

def cmd_validate_config(args: argparse.Namespace) -> int:
    errors = validate_config_file(args.config)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"Config valid: {args.config}")
    return 0


def cmd_prepare_data(args: argparse.Namespace) -> int:
    config, _ = load_config(args.config)
    config_hash_value = config_hash(args.config)
    frame, meta = load_evouna_samples(config)
    output_dir = prepare_output_dir(resolve_output_dir(config, config_hash_value))
    write_output_dir_manifest(output_dir, config, config_hash_value)
    frame.to_parquet(output_dir / "normalized_samples.parquet", index=False)
    write_resolved_config(output_dir / "config.resolved.yaml", resolved_config_with_redactions(args.config))
    (output_dir / "dataset_meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
    print(f"Prepared {len(frame)} normalized samples at {output_dir}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    config, _ = load_config(args.config)
    config_hash_value = config_hash(args.config)
    output_dir = prepare_output_dir(resolve_output_dir(config, config_hash_value))
    write_output_dir_manifest(output_dir, config, config_hash_value)
    normalized_path = output_dir / "normalized_samples.parquet"
    if normalized_path.exists():
        samples = pd.read_parquet(normalized_path)
        dataset_hash = ""
    else:
        samples, meta = load_evouna_samples(config)
        dataset_hash = meta["dataset_hash"]
        samples.to_parquet(normalized_path, index=False)
    if not dataset_hash:
        meta = yaml.safe_load((output_dir / "dataset_meta.yaml").read_text(encoding="utf-8"))
        dataset_hash = meta["dataset_hash"]
    write_resolved_config(output_dir / "config.resolved.yaml", resolved_config_with_redactions(args.config))
    parsed = run_predictions(
        config=config,
        normalized_samples=samples,
        output_dir=output_dir,
        config_hash_value=config_hash_value,
        dataset_hash=dataset_hash,
        resolved_config=resolved_config_with_redactions(args.config),
        resume=args.resume,
    )
    parsed.to_parquet(output_dir / "parsed_predictions.parquet", index=False)
    print(f"Completed predictions: {len(parsed)} rows")
    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    parsed = pd.read_parquet(output_dir / "parsed_predictions.parquet")
    write_metrics_bundle(parsed, output_dir, bootstrap_iterations=args.bootstrap_iterations)
    config_payload = yaml.safe_load((output_dir / "config.resolved.yaml").read_text(encoding="utf-8"))
    telemetry = config_payload.get("telemetry", {}) or {}
    if telemetry.get("enabled"):
        upload_file = build_metrics_dataset_file(
            output_dir / "metrics_overall.csv",
            output_dir / "parsed_predictions.parquet",
            output_dir,
            project_name=telemetry.get("project_name", "meta-judge-eval"),
            dataset_name=telemetry.get("dataset_name", "meta-judge-eval"),
        )
        dataset_status = sync_ax_dataset(
            telemetry.get("dataset_name", "meta-judge-eval"),
            upload_file,
            space=telemetry.get("space"),
            profile=telemetry.get("profile"),
        )
        manifest_path = output_dir / "telemetry_manifest.json"
        manifest = load_telemetry_manifest(manifest_path)
        if manifest:
            manifest.setdefault("ax", {})
            manifest["ax"]["dataset"] = dataset_status
            write_telemetry_manifest(manifest_path, manifest)
        if dataset_status.get("status") == "error":
            print(f"Failed to sync Arize dataset via ax: {dataset_status}", file=sys.stderr)
            return 1
    print(f"Metrics written to {output_dir}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    report_path = generate_report(Path(args.output_dir))
    print(f"Report written to {report_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="judge-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate-config")
    validate_parser.add_argument("config")
    validate_parser.set_defaults(func=cmd_validate_config)

    prepare_parser = subparsers.add_parser("prepare-data")
    prepare_parser.add_argument("config")
    prepare_parser.set_defaults(func=cmd_prepare_data)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("config")
    run_parser.add_argument("--resume", action="store_true")
    run_parser.set_defaults(func=cmd_run)

    metrics_parser = subparsers.add_parser("metrics")
    metrics_parser.add_argument("output_dir")
    metrics_parser.add_argument("--bootstrap-iterations", type=int, default=200)
    metrics_parser.set_defaults(func=cmd_metrics)

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("output_dir")
    report_parser.set_defaults(func=cmd_report)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
