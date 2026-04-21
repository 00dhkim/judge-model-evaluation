# 데이터셋 현황

기본 필터 기준: `improper=false`, `exclude_non_boolean_labels=true`, `exclude_empty_candidate_answers=true`

---

## EVOUNA/TQ.json

| 항목 | 값 |
|---|---|
| 원본 rows | 2,000 |
| proper rows | 1,938 |
| improper rows (필터 시 제외) | 62 |
| **필터링 후 총 samples** | **9,689** |

answer source별 sample 수:

| source | 개수 |
|---|---|
| chatgpt | 1,938 |
| fid | 1,937 |
| gpt35 | 1,938 |
| gpt4 | 1,938 |
| newbing | 1,938 |

---

## EVOUNA/NQ.json

| 항목 | 값 |
|---|---|
| 원본 rows | 3,610 |
| proper rows | 3,020 |
| improper rows (필터 시 제외) | 590 |
| **필터링 후 총 samples** | **15,094** |

answer source별 sample 수:

| source | 개수 |
|---|---|
| chatgpt | 3,020 |
| fid | 3,018 |
| gpt35 | 3,020 |
| gpt4 | 3,020 |
| newbing | 3,016 |

---

## 참고

- 1 row = 1 질문. 각 row마다 여러 answer source(chatgpt, fid, gpt35, gpt4, newbing)의 후보 답변이 있어, row 수 × source 수 ≈ 총 sample 수가 됩니다.
- fid / newbing은 일부 row에서 empty 또는 non-boolean label로 인해 제외되어 다른 source보다 1~4개 적습니다.
- 샘플 수를 제한하려면 `datasets[].sampling`을 사용하세요. 동일 시드 내 점진적 포함이 보장됩니다.
