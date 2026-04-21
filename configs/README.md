# Config 작성 가이드

실험 설정은 YAML 파일 한 개로 정의합니다. `uv run judge-eval run <config.yaml>` 으로 실행합니다.

## 파일 구조

```yaml
experiment_name: <string>       # 실험 식별자 (출력 디렉터리명과 일치시키는 것을 권장)
datasets: [...]                 # 평가 데이터셋 목록
filter: {...}                   # 데이터 필터 옵션
judge_models: [...]             # 평가할 judge 모델 목록
evaluation: {...}               # 평가 실행 옵션
output: {...}                   # 결과 저장 경로
telemetry: {...}                # Arize 텔레메트리 (선택)
```

---

## judge_models — provider 별 설정

### `openai_compatible` — OpenAI API 및 OpenAI 호환 서버

OpenAI API, Azure OpenAI, 또는 `/v1/chat/completions` 엔드포인트를 제공하는 서버에 사용합니다.

```yaml
judge_models:
  - name: gpt4o_judge
    provider: openai_compatible
    model: gpt-4o                                         # 필수
    endpoint: https://api.openai.com/v1/chat/completions  # 필수
    api_key_env: OPENAI_API_KEY                           # 환경변수 이름 (선택, 없으면 인증 생략)
    temperature: 0.0
    max_tokens: 256
```

**환경변수 설정 예시:**
```bash
export OPENAI_API_KEY=sk-...
```

`api_key_env` 에는 키 값이 아닌 **환경변수 이름**을 씁니다. `${VAR}` 문법으로도 참조 가능합니다.

---

### `vllm` — vLLM 서버

`vllm serve` 로 실행한 로컬/원격 서버에 사용합니다. 내부적으로 `openai_compatible` 과 동일한 Chat Completions 방식으로 호출됩니다.

```yaml
judge_models:
  - name: vllm_qwen
    provider: vllm
    model: Qwen/Qwen2.5-7B-Instruct   # 필수 — vllm serve 시 지정한 모델명
    endpoint: http://localhost:8000/v1/chat/completions  # 필수
    temperature: 0.0
    max_tokens: 256
```

**vLLM 서버 실행 예시:**
```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
```

---

### `openai_compatible` — Ollama (로컬 LLM)

Ollama는 `/v1/chat/completions` OpenAI 호환 API를 제공하므로 `openai_compatible` 을 사용합니다.

```yaml
judge_models:
  - name: ollama_llama3
    provider: openai_compatible
    model: llama3.2                                    # 필수 — ollama pull 한 모델명
    endpoint: http://localhost:11434/v1/chat/completions  # 필수
    temperature: 0.0
    max_tokens: 256
```

**Ollama 모델 준비:**
```bash
ollama pull llama3.2
ollama serve   # 이미 실행 중이면 생략
```

---

### `hf_local` — Hugging Face 로컬 파이프라인

`transformers` 패키지를 직접 사용해 GPU에서 실행합니다. 별도 서버 불필요.

```yaml
judge_models:
  - name: hf_phi3
    provider: hf_local
    model_path: microsoft/Phi-3-mini-4k-instruct  # model 또는 model_path 중 하나 필수
    temperature: 0.0
    max_tokens: 256
```

> `transformers` 패키지가 필요합니다: `uv add transformers torch`

---

### `custom_http` — 임의 HTTP 엔드포인트

`{"prompt": ..., "model": ..., "max_tokens": ...}` 를 POST하고 `{"content": ...}` 를 반환하는 커스텀 서버에 사용합니다.

```yaml
judge_models:
  - name: my_custom_server
    provider: custom_http
    endpoint: http://localhost:9000/generate  # 필수
    model: my-model
    max_tokens: 256
```

---

## evaluation 옵션

| 키 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `prompt_templates` | list | 전체 3종 | `minimal` / `guideline` / `guideline_with_examples` |
| `retry_count` | int | 1 | 파싱 실패 시 재시도 횟수 |
| `invalid_output_policy` | string | `store_invalid` | 파싱 불가 응답 처리 방식 |
| `bootstrap_iterations` | int | 200 | 신뢰구간 계산용 부트스트랩 반복 수 |
| `enable_prompt_sensitivity` | bool | true | 프롬프트 템플릿 간 일관성 분석 |
| `enable_reference_order_sensitivity` | bool | true | 정답/후보 순서 교환 민감도 분석 |
| `enable_dummy_answer_test` | bool | true | 더미 답변 편향 테스트 |

---

## 환경변수 참조 문법

YAML 값에서 `${ENV_VAR}` 형태로 환경변수를 참조할 수 있습니다.

```yaml
api_key_env: OPENAI_API_KEY   # 권장: 환경변수 이름을 문자열로 지정
space: ${ARIZE_SPACE_ID}      # 직접 치환
```

---

## 예시 파일

| 파일 | 설명 |
|---|---|
| `configs/examples/local_llm.yaml` | Ollama / vLLM / HF 로컬 LLM judge 예시 |
| `configs/examples/openai_judge.yaml` | OpenAI API (gpt-4o / gpt-4o-mini) judge 예시 |
| `configs/examples/full_evouna.yaml` | 전체 옵션을 활성화한 더미 모델 예시 |
