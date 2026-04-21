# judge-model-evaluation

EVOUNA-based LLM-as-a-Judge meta-evaluation toolkit.

## Quickstart

```bash
uv venv
uv sync
uv run judge-eval validate-config configs/examples/full_evouna.yaml
uv run judge-eval prepare-data configs/examples/full_evouna.yaml
uv run judge-eval run configs/examples/full_evouna.yaml --sample-size 50 --seed 42
uv run judge-eval metrics outputs/20260421_full_evouna_example
uv run judge-eval report outputs/20260421_full_evouna_example
uv run pytest -q
```

When `telemetry.enabled: true`, the runner ensures the Arize project via `ax` and exports OpenTelemetry spans to the configured Arize OTLP endpoint. The `metrics` command also materializes `arize_metrics_dataset.parquet` and syncs judge-level summary metrics to the Arize dataset named `meta-judge-eval` via `ax datasets create/append`.
