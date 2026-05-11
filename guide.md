# 운영 레퍼런스

## 명령어 목록

```bash
# 설정 파일 유효성 검사
uv run judge-eval validate-config <config.yaml>

# EVOUNA 데이터셋 로드 및 정규화, normalized_samples.parquet 생성
uv run judge-eval prepare-data <config.yaml>

# 각 judge 모델로 LLM 호출, parsed_predictions.parquet 생성
# base / prompt_sensitivity / reference_order / dummy_answer variant 포함
uv run judge-eval run <config.yaml>

# parse 실패 항목만 재실행
uv run judge-eval retry-failures <config.yaml>

# Scott's Pi, F1, FPR/FNR, leniency bias 등 지표 계산, CSV 생성
uv run judge-eval metrics <output_dir> [--exclude-models model1 model2]

# 지표 CSV로 모델 순위, 히트맵, 운영 적합성 리포트 생성
uv run judge-eval report <output_dir>

# 여러 실험 결과를 하나의 HTML 리포트로 병합
uv run judge-eval merge <dir1> <dir2> ... [--exclude-models ...] --output <out.html>
```

---

## 샘플 수 계산 방식

이 프로젝트에서는 세 가지 서로 다른 "개수"가 등장합니다.

### 1. 원본 질문 row 수

EVOUNA 원본 JSON은 질문 1개(row)마다 여러 candidate answer를 함께 담고 있습니다.  
예를 들어 질문 1개에 `answer_fid`, `answer_gpt35`, `answer_chatgpt`, `answer_gpt4`, `answer_newbing`이 함께 포함됩니다.

### 2. normalized sample 수 (prepare-data 이후)

`prepare-data`는 질문 row를 answer source별 평가 단위로 펼칩니다.  
질문 1개 × 유효한 answer source 5개 = normalized sample 5개.

config의 `datasets[].sampling.sample_size`는 이렇게 펼쳐진 뒤의 normalized sample 수를 지정합니다.

```yaml
# 이 설정은 질문 row 수가 아니라 normalized sample 수
datasets:
  - name: evouna_tq
    sampling:
      sample_size: 500
  - name: evouna_nq
    sampling:
      sample_size: 500
# → run 입력: normalized sample 1000개
```

### 3. evaluation row 수 (run 실행 시)

`run`은 normalized sample 1개를 다시 여러 evaluation row로 복제합니다.

| variant | 추가 row 수 |
|---|---|
| base | 1 |
| prompt_sensitivity | prompt template 수 (예: 3) |
| reference_order_sensitivity | 최대 3 (alias 수에 따라) |
| dummy_answer_test | 5 |

prompt template 3개, 모든 variant 활성화 시 normalized sample 1개 → 최소 9개 이상의 evaluation row.

```
sample_size 1000 × ~10 variants = 진행 표시에 약 10,000개
실제 호출 수 = evaluation row 수 × judge model 수
```

---

## Resume vs Retry

| 명령어 | 동작 |
|---|---|
| `run --resume` | raw에 아직 없는 `unit_key`만 채운다 |
| `retry-failures` | 현재 raw의 최종 상태가 `error` 또는 `invalid`인 `unit_key`만 재실행 |

둘 다 `raw_predictions.jsonl`은 append-only로 유지하며,  
`parsed_predictions.parquet`은 전체 raw에서 `unit_key`별 마지막 레코드 기준으로 재생성됩니다.

---

## Telemetry (Arize)

`telemetry.enabled: true` 설정 시:

- runner가 Arize 프로젝트를 `ax`로 확인하고 OpenTelemetry span을 OTLP 엔드포인트로 전송합니다.
- `metrics` 명령어가 `arize_metrics_dataset.parquet`을 생성하고, judge 수준 요약 지표를 Arize 데이터셋 `meta-judge-eval`에 동기화합니다.
