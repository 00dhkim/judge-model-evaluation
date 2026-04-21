# judge-model-evaluation

EVOUNA-based LLM-as-a-Judge meta-evaluation toolkit.

## Quickstart

```bash
uv venv
uv sync

# YAML 설정 파일의 필드와 값을 파싱해 유효성을 검사한다. 오류가 있으면 stderr에 출력하고 종료한다.
uv run judge-eval validate-config configs/examples/full_evouna.yaml

# EVOUNA 데이터셋을 로드하고 필터 정책을 적용해 정규화한 뒤, normalized_samples.parquet으로 저장한다.
uv run judge-eval prepare-data configs/examples/full_evouna.yaml

# 정규화된 샘플에 대해 각 judge 모델로 LLM 호출을 수행하고, 응답을 파싱해 parsed_predictions.parquet으로 저장한다.
# base 평가 외에 prompt sensitivity·reference order·dummy answer 등 variant 평가도 함께 실행한다.
uv run judge-eval run configs/examples/full_evouna.yaml --sample-size 50 --seed 42

# parsed_predictions.parquet을 읽어 Scott's Pi, F1, precision/recall, FPR/FNR, leniency bias 등
# 다양한 지표를 계산하고, 모델·데이터셋·답변 길이 등 여러 축으로 분류한 CSV 파일들을 저장한다.
uv run judge-eval metrics outputs/20260421_full_evouna_example

# 계산된 지표 CSV들을 읽어 모델 순위·약점·운영 적합성 요약을 담은 마크다운 리포트와
# Scott's Pi, precision/recall, prompt sensitivity 히트맵 등 시각화 차트를 생성한다.
uv run judge-eval report outputs/20260421_full_evouna_example

uv run pytest -q
```

When `telemetry.enabled: true`, the runner ensures the Arize project via `ax` and exports OpenTelemetry spans to the configured Arize OTLP endpoint. The `metrics` command also materializes `arize_metrics_dataset.parquet` and syncs judge-level summary metrics to the Arize dataset named `meta-judge-eval` via `ax datasets create/append`.
