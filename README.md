# judge-model-evaluation

EVOUNA-based LLM-as-a-Judge meta-evaluation toolkit.

## Quickstart

```bash
uv venv
uv sync

# YAML 설정 파일의 필드와 값을 파싱해 유효성을 검사한다. 오류가 있으면 stderr에 출력하고 종료한다.
uv run judge-eval validate-config configs/examples/gemma4.yaml

# EVOUNA 데이터셋을 로드하고 필터 정책을 적용해 정규화한 뒤, normalized_samples.parquet으로 저장한다.
uv run judge-eval prepare-data configs/examples/gemma4.yaml

# 정규화된 샘플에 대해 각 judge 모델로 LLM 호출을 수행하고, 응답을 파싱해 parsed_predictions.parquet으로 저장한다.
# base 평가 외에 prompt sensitivity·reference order·dummy answer 등 variant 평가도 함께 실행한다.
# prepare-data를 먼저 실행했다면, run은 기존 output 디렉터리를 재사용한다. 날짜가 바뀌어도 같은 config_hash면 이어서 쓴다.
uv run judge-eval run configs/examples/gemma4.yaml

# parsed_predictions.parquet을 읽어 Scott's Pi, F1, precision/recall, FPR/FNR, leniency bias 등
# 다양한 지표를 계산하고, 모델·데이터셋·답변 길이 등 여러 축으로 분류한 CSV 파일들을 저장한다.
uv run judge-eval metrics outputs/20260421_gemma4_v1

# 계산된 지표 CSV들을 읽어 모델 순위·약점·운영 적합성 요약을 담은 마크다운 리포트와
# Scott's Pi, precision/recall, prompt sensitivity 히트맵 등 시각화 차트를 생성한다.
uv run judge-eval report outputs/20260421_gemma4_v1

uv run pytest -q
```

## How Sample Counts Work

이 프로젝트에는 서로 다른 세 가지 개수가 있습니다.

1. 원본 질문 row 수
2. `prepare-data` 이후의 normalized sample 수
3. `run`에서 실제로 평가되는 evaluation row 수

EVOUNA 원본 JSON은 질문 1개(row)마다 여러 candidate answer를 함께 담고 있습니다. 예를 들어 질문 1개 안에 `answer_fid`, `answer_gpt35`, `answer_chatgpt`, `answer_gpt4`, `answer_newbing`가 같이 들어 있습니다.

`prepare-data`는 질문 row를 answer source별 평가 단위로 펼칩니다. 그래서 질문 1개는 최대 5개의 normalized sample이 됩니다.

예를 들어:

- 질문 row 1개
- 유효한 answer source가 5개

이면 `prepare-data` 결과는 normalized sample 5개입니다.

따라서 config의 `datasets[].sampling.sample_size`는 "질문 몇 개를 뽑을지"가 아니라, 이렇게 펼쳐진 뒤의 normalized sample을 몇 개 남길지 지정하는 값입니다.

예를 들어:

- `evouna_tq`에서 `sample_size: 500`
- `evouna_nq`에서 `sample_size: 500`

이면 `run`에 들어가는 기본 입력은 질문 1000개가 아니라 normalized sample 1000개입니다.

여기서 끝이 아닙니다. `run`은 normalized sample 1개를 다시 여러 evaluation row로 복제합니다.

- `base`: 1개
- `prompt_sensitivity`: prompt template 수만큼 추가
- `reference_order_sensitivity`: golden answer alias가 여러 개면 최대 3개 추가
- `dummy_answer_test`: 5개 추가

기본 예시 설정처럼 prompt template이 3개이고, variant 옵션이 모두 켜져 있으면 normalized sample 1개는 보통 9개 이상으로 늘어납니다.

- 최소: `1(base) + 3(prompt) + 5(dummy) = 9`
- alias reorder가 가능한 샘플은 여기에 `reference_order`가 1~3개 더 붙음

그래서:

- `sample_size: 500 + 500 = 1000`

으로 설정해도, `run` 진행 표시에는 약 10500개의 samples가 잡힐 수 있습니다. 이 수치는 질문 row 수가 아니라, variant까지 확장된 실제 evaluation row 수입니다.

실제 모델 호출 수는 여기에 judge model 수가 다시 곱해집니다.

- `실제 호출 수 = evaluation row 수 × judge model 수`

When `telemetry.enabled: true`, the runner ensures the Arize project via `ax` and exports OpenTelemetry spans to the configured Arize OTLP endpoint. The `metrics` command also materializes `arize_metrics_dataset.parquet` and syncs judge-level summary metrics to the Arize dataset named `meta-judge-eval` via `ax datasets create/append`.
