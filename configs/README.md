# Config 작성 가이드

실험 설정은 YAML 파일 한 개로 정의합니다.

```bash
uv run judge-eval run configs/my_experiment.yaml
```

---

## 최상위 필드

| 필드 | 타입 | 필수 | 설명 |
|---|---|---|---|
| `experiment_name` | string | 필수 | 실험 식별자. 출력 디렉터리명과 일치시키는 것을 권장합니다. |
| `datasets` | list | 필수 | 평가에 사용할 데이터셋 목록. 최소 1개 필요. |
| `filter` | object | 선택 | 데이터 필터 옵션. 생략 시 기본값 적용. |
| `judge_models` | list | 필수 | 평가할 judge 모델 목록. 최소 1개 필요. |
| `evaluation` | object | 선택 | 평가 실행 옵션. 생략 시 기본값 적용. |
| `output` | object | 필수 | 결과 저장 경로 및 옵션. |
| `telemetry` | object | 선택 | Arize 텔레메트리 연동. 생략 시 비활성화. |

---

## `datasets[]`

데이터셋은 여러 개를 지정할 수 있으며, 각 데이터셋마다 독립적으로 평가가 수행됩니다.

| 필드 | 타입 | 필수 | 설명 |
|---|---|---|---|
| `name` | string | 필수 | 데이터셋 식별자. 출력 파일의 `dataset` 컬럼에 기록됩니다. |
| `path` | string | 필수 | JSON 파일 경로. 프로젝트 루트 기준 상대경로 또는 절대경로. |
| `sampling` | object | 선택 | 이 데이터셋에만 적용할 샘플링 설정. 최상위 `sampling`보다 우선합니다. 데이터 개수에 따른 점진적 포함을 보장합니다. |

```yaml
datasets:
  - name: evouna_tq
    path: data/EVOUNA/TQ.json
    sampling:
      sample_size: 500
      seed: 42
  - name: evouna_nq
    path: data/EVOUNA/NQ.json
    sampling:
      sample_size: 1000
      seed: 42
```

---

## `filter`

데이터셋에서 불량 샘플을 제거하는 필터입니다.

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `improper` | bool | `false` | `true`로 설정하면 improper 플래그가 붙은 샘플을 **포함**합니다. `false`이면 제외합니다. |
| `exclude_non_boolean_labels` | bool | `true` | true/false 외의 레이블을 가진 샘플을 제외합니다. 강력 권장값은 `true`입니다. `false`이면 non-boolean 레이블이 경고와 함께 `False`로 강제 변환될 수 있습니다. |
| `exclude_empty_candidate_answers` | bool | `true` | candidate answer가 비어있는 샘플을 제외합니다. |

```yaml
filter:
  improper: false
  exclude_non_boolean_labels: true
  exclude_empty_candidate_answers: true
```

---

## `judge_models[]`

평가할 judge 모델 목록입니다. 나열된 순서대로 각 샘플에 대해 순차 호출됩니다.

### 공통 필드

| 필드 | 타입 | 필수 | 기본값 | 설명 |
|---|---|---|---|---|
| `name` | string | 필수 | — | 모델 식별자. 출력 파일의 `judge_model` 컬럼에 기록됩니다. |
| `provider` | string | 필수 | — | 호출 방식. 아래 provider 목록 참조. |
| `model` | string | provider에 따라 | — | 모델명. `openai_compatible`, `vllm`에서 필수. |
| `model_path` | string | provider에 따라 | — | 로컬 모델 경로 또는 HuggingFace ID. `hf_local`에서 `model`과 택일. |
| `endpoint` | string | provider에 따라 | — | HTTP 엔드포인트 URL. `openai_compatible`, `vllm`, `custom_http`에서 필수. |
| `api_key_env` | string | 선택 | — | API 키를 담은 **환경변수 이름**. 값 자체가 아닌 변수명을 씁니다. |
| `temperature` | float | 선택 | `0.0` | 샘플링 온도. judge 평가는 `0.0` 권장. |
| `max_tokens` | int | 선택 | `256` | 생성 최대 토큰 수. |
| `metadata` | object | 선택 | `{}` | 임의 키-값. `dummy` provider의 `dummy_strategy` 등에 사용. |

### Provider 종류

#### `openai_compatible` — OpenAI API 및 OpenAI 호환 서버

`/v1/chat/completions` 엔드포인트를 제공하는 모든 서버에 사용합니다 (OpenAI, Azure OpenAI, Ollama 등).

**필수 필드:** `model`, `endpoint`

```yaml
- name: gpt4o_judge
  provider: openai_compatible
  model: gpt-4o
  endpoint: https://api.openai.com/v1/chat/completions
  api_key_env: OPENAI_API_KEY
  temperature: 0.0
  max_tokens: 256
```

> Rate limit(HTTP 429) 발생 시 `Retry-After` 헤더 기반 exponential backoff를 최대 10회 자동 재시도합니다.

---

#### `vllm` — vLLM 서버

`vllm serve`로 실행한 로컬/원격 서버에 사용합니다. 내부적으로 `openai_compatible`과 동일한 Chat Completions API를 호출합니다.

**필수 필드:** `model`, `endpoint`

```yaml
- name: vllm_qwen
  provider: vllm
  model: Qwen/Qwen2.5-7B-Instruct
  endpoint: http://localhost:8000/v1/chat/completions
  temperature: 0.0
  max_tokens: 256
```

**vLLM 서버 실행:**
```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
```

---

#### `hf_local` — Hugging Face 로컬 파이프라인

`transformers` 패키지를 직접 사용해 로컬 GPU에서 실행합니다. 별도 서버 불필요.

**필수 필드:** `model` 또는 `model_path` 중 하나

```yaml
- name: hf_phi3
  provider: hf_local
  model_path: microsoft/Phi-3-mini-4k-instruct
  temperature: 0.0
  max_tokens: 256
```

> `transformers`와 `torch` 패키지가 필요합니다: `uv add transformers torch`
>
> 동일한 `hf_local` 모델은 프로세스 내에서 한 번만 로드되고 이후 호출에서 재사용됩니다.

---

#### `custom_http` — 임의 HTTP 엔드포인트

`{"prompt": ..., "model": ..., "max_tokens": ...}`를 POST하고 `{"content": ...}` (또는 `text`, `output`)를 반환하는 커스텀 서버에 사용합니다.

**필수 필드:** `endpoint`

```yaml
- name: my_server
  provider: custom_http
  endpoint: http://localhost:9000/generate
  model: my-model
  max_tokens: 256
```

응답 JSON에서 `content` → `text` → `output` 순으로 문자열 필드를 탐색합니다. 추가로 `input_tokens`, `output_tokens`, `estimated_cost` 필드를 포함하면 메트릭에 반영됩니다.

---

#### `dummy` — 더미 모델 (테스트용)

실제 LLM을 호출하지 않고 규칙 기반으로 판단합니다. 파이프라인 동작 검증에 사용합니다.

```yaml
- name: perfect_dummy
  provider: dummy
  metadata:
    dummy_strategy: heuristic   # heuristic | perfect | always_true | always_false | invalid
    model_family: dummy
```

| `dummy_strategy` | 동작 |
|---|---|
| `heuristic` (기본) | golden answer와 candidate answer의 부분 문자열 일치 여부로 판단 |
| `perfect` | 정확한 포함 여부로 판단 |
| `always_true` | 항상 `true` 반환 |
| `always_false` | 항상 `false` 반환 |
| `invalid` | 파싱 불가 응답 반환 (invalid output 처리 경로 테스트용) |

---

## `evaluation`

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `prompt_templates` | list[string] | 전체 3종 | judge에게 전달할 프롬프트 템플릿. 아래 참조. |
| `retry_count` | int | `1` | 응답 파싱 실패(invalid JSON 등) 시 재시도 횟수. rate limit 재시도와 별개입니다. |
| `invalid_output_policy` | string | `store_invalid` | 파싱 불가 응답 처리 방식. 현재 `store_invalid`만 지원. |
| `bootstrap_iterations` | int | `200` | Scott's Pi 등 신뢰구간 계산용 부트스트랩 반복 횟수. |
| `enable_prompt_sensitivity` | bool | `true` | 프롬프트 템플릿 변경 시 판단이 달라지는지 분석합니다. |
| `enable_reference_order_sensitivity` | bool | `true` | golden answer alias 순서를 바꿨을 때 판단 변화를 분석합니다. |
| `enable_dummy_answer_test` | bool | `true` | 자명한 더미 답변(정답 그대로, 빈 문자열 등)에 대한 편향을 분석합니다. |

### `prompt_templates` 종류

| 값 | 설명 |
|---|---|
| `minimal` | 질문·정답·후보만 제시하는 최소 프롬프트 |
| `guideline` | 판단 기준(가이드라인)을 포함한 프롬프트 |
| `guideline_with_examples` | 가이드라인 + few-shot 예시를 포함한 프롬프트 |

```yaml
evaluation:
  prompt_templates:
    - minimal
    - guideline
    - guideline_with_examples
  retry_count: 1
  bootstrap_iterations: 200
  enable_prompt_sensitivity: true
  enable_reference_order_sensitivity: true
  enable_dummy_answer_test: true
```

---

## `output`

| 필드 | 타입 | 필수 | 기본값 | 설명 |
|---|---|---|---|---|
| `experiment_name` | string | 필수 | — | 실험 식별자. 출력 디렉터리 이름으로 사용됩니다. |
| `base_dir` | string | 선택 | `"outputs"` | 출력 루트 디렉터리. 실제 경로는 `{base_dir}/{experiment_name}`. |
| `save_raw_predictions` | bool | 선택 | `true` | `raw_predictions.jsonl` 저장 여부. |
| `save_report` | bool | 선택 | `true` | `report.md` 및 `plots/` 저장 여부. |

실행 당일 날짜가 자동으로 prefix로 붙습니다. 예를 들어 `experiment_name: my_experiment`로 설정하면 실제 출력 디렉터리는 `outputs/20260421_my_experiment`가 됩니다. 다만 `prepare-data`가 이미 출력 디렉터리를 materialize했다면, 이후 `run`은 날짜가 바뀌어도 같은 config hash의 기존 디렉터리를 재사용합니다.

```yaml
output:
  experiment_name: my_experiment   # 실제 경로: outputs/{YYYYMMDD}_my_experiment
  # base_dir: outputs              # 기본값. 변경 시 {base_dir}/{YYYYMMDD}_my_experiment
  save_raw_predictions: true
  save_report: true
```

### 생성되는 출력 파일

| 파일 | 설명 |
|---|---|
| `raw_predictions.jsonl` | 모든 LLM 호출의 원본 응답 (JSONL) |
| `parsed_predictions.parquet` | 파싱된 예측값 및 메타데이터 |
| `metrics_overall.csv` | 모델×프롬프트 조합별 Scott's Pi, F1, FPR, FNR 등 |
| `metrics_by_answer_source.csv` | 답변 출처(fid, gpt4 등)별 분해 메트릭 |
| `model_rankings.csv` | 지표별 모델 순위 |
| `prompt_sensitivity.csv` | 프롬프트 간 label flip rate |
| `dummy_answer_robustness.csv` | 더미 답변 유형별 정확도 |
| `leniency_bias.csv` | judge의 과대/과소 평가 경향 |
| `report.md` | 전체 결과 요약 리포트 (마크다운) |
| `plots/` | Scott's Pi, FPR/FNR, 프롬프트 민감도 히트맵 등 시각화 |
| `resolved_config.yaml` | 환경변수가 치환된 최종 설정 스냅샷 |
| `telemetry_manifest.json` | Arize 텔레메트리 전송 내역 |

---

## `telemetry`

Arize Phoenix로 trace를 전송합니다. `enabled: false`이면 나머지 필드는 무시됩니다.

| 필드 | 타입 | 필수 | 기본값 | 설명 |
|---|---|---|---|---|
| `enabled` | bool | — | `false` | `true`로 설정하면 Arize로 trace를 전송합니다. |
| `provider` | string | enabled=true일 때 필수 | `null` | 현재 `arize`만 지원합니다. |
| `project_name` | string | 선택 | `"meta-judge-eval"` | Arize 프로젝트 이름. |
| `dataset_name` | string | 선택 | `"meta-judge-eval"` | Arize 데이터셋 이름. |
| `space` | string | 선택 | — | Arize Space ID. `${ARIZE_SPACE_ID}` 환경변수 참조 권장. |
| `profile` | string | 선택 | — | Arize 프로파일 이름. |

`enabled: true` 사용 시 다음 환경변수가 반드시 설정되어 있어야 합니다:

```bash
export ARIZE_API_KEY=...
export ARIZE_SPACE_ID=...
```

```yaml
telemetry:
  enabled: true
  provider: arize
  project_name: my-project
  dataset_name: my-dataset
  space: ${ARIZE_SPACE_ID}
```

---

## 환경변수 참조 문법

YAML 값에서 `${ENV_VAR}` 형태로 환경변수를 참조할 수 있습니다. 실행 시점에 치환됩니다.

```yaml
api_key_env: OPENAI_API_KEY     # 권장: 변수명을 문자열로 지정 (api_key_env 전용)
space: ${ARIZE_SPACE_ID}        # 직접 치환 (모든 문자열 필드에 사용 가능)
```

환경변수가 설정되지 않은 경우 해당 값은 `null`로 처리됩니다. `telemetry.enabled: true` 상태에서 필수 환경변수가 누락되면 실행 시 오류가 발생합니다.

---

## 예시 파일

| 파일 | 설명 |
|---|---|
| [`configs/examples/local_llm.yaml`](examples/local_llm.yaml) | Ollama / vLLM / HF 로컬 LLM judge 예시 |
| [`configs/examples/openai_judge.yaml`](examples/openai_judge.yaml) | OpenAI API (gpt-4o / gpt-4o-mini) judge 예시 |
| [`configs/examples/full_evouna.yaml`](examples/full_evouna.yaml) | 전체 옵션을 활성화한 더미 모델 예시 |
