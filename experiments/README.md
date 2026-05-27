# Judge Model Optimization Experiments

Solar Pro3 및 K-EXAONE-236B-A23B judge 모델의 Scott's π를 개선하기 위한 프롬프트·추론 전략 실험.

## 배경

| 모델 | Scott's π | 비고 |
|------|-----------|------|
| solar_pro_3_tuned (baseline) | 0.6991 | 이 실험의 출발점 |
| k_exaone_236b_a23b_tuned (baseline) | 0.6845 | |
| gpt-oss-120b | 0.7240 | **목표 하한** |
| gemma-4-26b | 0.8210 | **목표 상한** |

평가 데이터: EVOUNA TQ 150샘플 + NQ 150샘플 = 300샘플 (스크리닝), 1200샘플 (final)

---

## 최적화 방법론

### Tier 1 — 코드 수정 거의 없이 즉시 효과 (예상 +0.02 ~ +0.05)

#### m1 — Self-Consistency (다중 샘플링 + 다수결)

가장 ROI 큰 단일 기법. CoT를 사용하는 judge에서 K=5~10 샘플 후 majority vote만 해도 GSM8K +17.9%, NDCG +7.5pt 수준의 일관된 이득이 보고됨. CISC(Confidence-weighted Self-Consistency) 변형은 K=10으로 K=18.6 수준 성능 달성.  
([Confidence Improves Self-Consistency in LLMs](https://arxiv.org/pdf/2502.06233), [Self-Consistency Sampling](https://www.emergentmind.com/topics/self-consistency-sampling))

temperature=0.7(이미 우리 세팅)로 N=5 호출, JSON label 다수결, 동률 시 confidence(이유 길이/일관성)로 가중. Solar Pro3 cached_input $0.015/M라 비용 부담 적음.

#### m2 — CLEV-style Alias Enumeration (Reference-Guided Verdict)

EVOUNA 같은 short-form QA에서 lexical alias match를 사전 필터로 사용 → LLM은 모호한 케이스만 판정. 정확도 유지하면서 noise/cost 큰 폭 감소. Reference-Guided Verdict는 golden+aliases 명시화로 false negative 큰 감소 보고.  
([CLEV](https://arxiv.org/html/2503.08542), [Reference-Guided Verdict](https://arxiv.org/html/2408.09235v3))

prompt에 `Acceptable forms: [alias1, alias2, ...]` 명시. EVOUNA TQ/NQ는 이미 golden_answers에 `/` 구분된 별칭이 들어있어서 그대로 enumerate하면 됨.

#### m3 — Alias Shuffle (Order Bias 대응)

JudgeLM의 핵심 트릭을 alias 순서에 적용. golden ↔ candidate alias 순서를 두 가지 랜덤 시드로 섞어 2회 호출, 일치할 때 채택, 불일치 시 원래 순서로 3번째 호출 → 다수결(2/3). Position bias가 큰 모델일수록 이득이 큼.  
([JudgeLM ICLR 2025](https://arxiv.org/pdf/2310.17631))

EVOUNA reference_order_sensitivity 결과가 이미 측정 가능 — 차이가 큰 모델일수록 swap voting 이득 큼.

#### m4 — Explicit CoT 강화

"Explicit Reasoning Makes Better Judges" (2025)에서 thinking mode가 7-shot ICL 대비 +10.5pt vs +4.5pt, 비용은 8.16× → 1.82×로 효율적. Solar Pro3는 이미 reasoning_effort: high이며, EXAONE은 reasoning 토큰을 명시적으로 요구하는 시스템 프롬프트가 없음.  
([Explicit Reasoning Makes Better Judges](https://arxiv.org/pdf/2509.13332))

EXAONE에 `<think>...</think>` 또는 명시적 "step 1/2/3" 구조 시스템 프롬프트 주입. K-EXAONE-236B-A23B는 MoE라 reasoning 토큰 늘릴 여지 큼.  
⚠️ **Solar Pro3에서 역효과** (π -0.087): reasoning_effort=high와 충돌 추정.

---

### Tier 2 — 중간 구현 비용 (예상 추가 +0.03 ~ +0.06)

#### m5 — Decomposed/Rubric Judging

하나의 boolean 판정을 다단계로 쪼갬: (a) 핵심 entity 추출 → (b) alias normalization → (c) semantic equivalence → (d) final label. 각 sub-step을 작은 LLM 콜로 분리하면 약한 모델일수록 효과 큼. eugeneyan 정리 및 SAGE의 local consistency 개념.  
([Eugene Yan — LLM-evaluators](https://eugeneyan.com/writing/llm-evaluators/), [SAGE/D3](https://arxiv.org/pdf/2410.04663))

⚠️ **Solar Pro3에서 파싱 실패 다수** (coverage 64%, π 0.22): JSON 포맷 불안정. Solar에는 적합하지 않음.

#### m6 — Few-shot 예시 큐레이션 강화 (Extended Few-shot)

현재 guideline_with_examples 템플릿의 예시 수/품질을 늘림. Mashee 결과: 충분한 ICL 예시 시 exact match 0.879, κ 0.807까지. ICL 단독은 reasoning 대비 효율 낮지만, short-form QA judge에서 자주 헷갈리는 일반 패턴(별칭 차이, 동의어, 부분 일치, 잘못된 entity)을 수동 예시로 명시하면 +0.01~0.03 기대.  
([LLMs-as-Judges Survey](https://arxiv.org/html/2412.05579v2))

구현상 extended few-shot 8개는 EVOUNA 실제 오답 케이스에서 샘플링한 항목이 아니라, 일반적인 판정 패턴을 설명하기 위해 수동 작성한 예시다. 코드 기준 해당 8개 golden/candidate 쌍은 EVOUNA TQ/NQ의 실제 평가 행과 정확히 일치하지 않는다. 단, `Paris`, `Tokyo`, `United States`처럼 예시에 쓰인 일부 엔티티 문자열 자체는 EVOUNA에도 등장하므로, test-set에서 추출한 hard-negative라고 주장하지 않는다.

#### m7 — Confidence Calibration + Abstain

모델에 confidence 수치(1–5점 또는 확률)를 요청해서 threshold 미만이면 abstain, 그렇지 않으면 채택. Scott's π는 일관된 응답에 보상이 크므로 confident 케이스만 채택해도 π 상승. SAGE 분석과 일치.  
([SAGE](https://www.emergentmind.com/topics/llm-as-a-judge-methodology))

**개별 방법 중 최고 성능** (스크리닝 π 0.7483, +0.049). 조합 실험에서 sc5 + ext_fewshot과 결합 시 추가 향상.

---

## 실험 결과

### 개별 메서드 스크리닝 결과 (300샘플)

| 순위 | Method | π | Δ vs baseline | coverage | 비고 |
|------|--------|---|---|---|---|
| 1 | m7_confidence_abstain | **0.7483** | +0.049 | 100% | ★ gpt-oss-120b 초과 |
| 2 | m3_alias_shuffle | 0.7409 | +0.042 | 100% | ★ |
| 3 | m1_sc_n5 | 0.7373 | +0.038 | 100% | ★ |
| 4 | m6_extended_fewshot | 0.7118 | +0.013 | 99% | |
| 5 | m2_alias_enum | 0.6762 | -0.023 | 98% | 단독 사용 시 역효과 |
| 6 | m4_explicit_cot | 0.6124 | -0.087 | 95% | ⚠️ Solar과 충돌 |
| 7 | m5_decomposed | 0.2169 | -0.482 | 64% | ⚠️ JSON 파싱 실패 |

★ = gpt-oss-120b 목표(0.724) 초과  
목표 상한: gemma-4-26b π=0.821

### 조합 실험 스크리닝 결과 (300샘플)

#### Round 1 — 사전 설계

| ID | 구성 | π | Δ |
|----|------|---|---|
| c01_sc5_aliasenum | sc5 + alias_enum | 0.7463 | +0.047 ★ |
| c02_sc5_extfewshot | sc5 + ext_fewshot | 0.7335 | +0.034 ★ |
| c03_aliasenum_extfewshot | alias_enum + ext_fewshot | 0.6918 | -0.007 |
| c04_sc5_aliasenum_extfewshot | sc5 + alias_enum + ext_fewshot | 0.7294 | +0.030 ★ |
| c05_decomposed_extfewshot | decomposed + ext_fewshot | 0.3216 | -0.378 | ⚠️ coverage 61% |
| c07_sc5_aliasshuffle_extfewshot | sc5 + alias_shuffle + ext_fewshot | 0.7443 | +0.045 ★ |
| c10_all_lite | sc3 + alias_enum + ext_fewshot + confidence | 0.7708 | +0.072 ★ |

#### Round 2 — m7(confidence_abstain) 중심 조합

개별 실험에서 m7(+0.049), m3(+0.042), m1(+0.038)이 상위 3개로 확인. decomposed(m5), CoT(m4)는 역효과 확인으로 조합에서 제외.

| ID | 구성 | π | Δ |
|----|------|---|---|
| n06_sc5_ext_confidence | sc5 + ext_fewshot + confidence | **0.7801** | +0.081 ★ |
| c10_all_lite | sc3 + alias_enum + ext_fewshot + confidence | 0.7708 | +0.072 ★ |
| n01_sc5_confidence | sc5 + confidence | 0.7690 | +0.070 ★ |
| n05_sc3_shuffle_confidence | sc3 + alias_shuffle + confidence | 0.7519 | +0.053 ★ |
| m7_confidence_abstain | confidence (단독) | 0.7483 | +0.049 ★ |
| c01_sc5_aliasenum | sc5 + alias_enum | 0.7463 | +0.047 ★ |
| c07_sc5_aliasshuffle_extfewshot | sc5 + alias_shuffle + ext_fewshot | 0.7443 | +0.045 ★ |
| n04_sc5_shuffle_ext_confidence | sc5 + alias_shuffle + ext + confidence | 0.7199 | +0.021 ★ |
| n02_shuffle_confidence | alias_shuffle + confidence | 0.7129 | +0.013 ★ |
| n03_sc5_shuffle_confidence | sc5 + alias_shuffle + confidence | 0.7125 | +0.013 ★ |
| n07_shuffle_ext_confidence | alias_shuffle + ext_fewshot + confidence | 0.7052 | +0.006 |

---

## Final 실험 결과 (1200샘플)

스크리닝 π 기준 상위 5개를 1200샘플(TQ 600 + NQ 600)로 재실행.

| 순위 | Method | π (1200샘플) | Δ vs baseline | 비고 |
|------|--------|-------------|---------------|------|
| 1 | n06_sc5_ext_confidence | **0.8016** | +0.103 | ★★ 목표 상한 근접 |
| 2 | n01_sc5_confidence | **0.7950** | +0.096 | ★★ |
| 3 | c10_all_lite | 0.7694 | +0.070 | ★ |
| 4 | n05_sc3_shuffle_confidence | 0.7575 | +0.058 | ★ |
| 5 | m7_confidence_abstain | 0.7317 | +0.033 | ★ |

★ = gpt-oss-120b(0.724) 초과 / ★★ = 목표 상한 gemma-4-26b(0.821) 근접  
baseline: Solar Pro3 0.6991

**핵심 결론**: sc5(Self-Consistency) + ext_fewshot(8개 hard-negative 예시) + confidence_abstain 조합(n06)이 π 0.8016으로 최고 성능. gemma-4-26b(0.821) 대비 0.019p 차이.

---

## K-EXAONE Final 실험 결과 (1200샘플)

K-EXAONE-236B-A23B (Friendli 서버리스, reasoning_budget=2048) 기준, 동일 5개 방법을 1200샘플(TQ 600 + NQ 600)로 재실행.

| 순위 | Method | π (1200샘플) | Δ vs baseline | 비고 |
|------|--------|-------------|---------------|------|
| 1 | c10_all_lite | **0.8340** | +0.150 | ★★ 목표 상한 초과 |
| 2 | n05_sc3_shuffle_confidence | 0.7982 | +0.114 | ★★ |
| 3 | n06_sc5_ext_confidence | 0.7831 | +0.099 | ★★ |
| 4 | n01_sc5_confidence | 0.7455 | +0.061 | ★ |
| 5 | m7_confidence_abstain | 0.6807 | -0.004 | baseline 미달 |

★ = gpt-oss-120b(0.724) 초과 / ★★ = 목표 상한 gemma-4-26b(0.821) 초과  
baseline: K-EXAONE 0.6845

**핵심 결론**: c10_all_lite(sc3 + alias_enum + ext_fewshot + confidence) 조합이 π 0.8340으로 최고 성능. gemma-4-26b(0.821) 목표 상한을 초과. m7(confidence abstain 단독)은 EXAONE에서 역효과(-0.004) — Solar와 달리 confidence 기반 기권 전략이 맞지 않음.

---

## 집계 산출물 원칙

최적화 실험의 원천 자료는 각 메서드 디렉터리의 `predictions.jsonl`이다. `summary.csv` / `summary_final.csv`는 해당 실행에서 돌린 메서드 subset만 담을 수 있으므로, 발표·보고용 전체 리더보드는 전체 디렉터리를 다시 스캔하는 `summary_full.csv`를 기준으로 한다.

```bash
uv run python scripts/aggregate_optim_results.py --root outputs/optim_final --judge solar
uv run python scripts/aggregate_optim_results.py --root outputs/optim_final_exaone --judge exaone
```

위 명령은 하위 `*/predictions.jsonl` 전체를 `normalized_samples.parquet`와 `sample_id`로 병합해 `summary_full.csv`를 재생성하고, 읽은 파일 목록과 sample hash를 `summary_manifest.json`에 기록한다.

---

## 출처

- [JudgeLM (ICLR 2025 Spotlight)](https://arxiv.org/pdf/2310.17631) / [GitHub](https://github.com/baaivision/JudgeLM)
- [Explicit Reasoning Makes Better Judges (2025)](https://arxiv.org/pdf/2509.13332)
- [Confidence Improves Self-Consistency (ACL 2025)](https://aclanthology.org/2025.findings-acl.1030.pdf)
- [CLEV — Lightweight Efficient Voting for Free-Form QA](https://arxiv.org/html/2503.08542)
- [Reference-Guided Verdict](https://arxiv.org/html/2408.09235v3)
- [LLMs-as-Judges Survey](https://arxiv.org/html/2412.05579v2)
- [LLM-as-Judge — Eugene Yan](https://eugeneyan.com/writing/llm-evaluators/)
- [Evaluating Open-QA Evaluation (EVOUNA, Amazon Science)](https://assets.amazon.science/86/1a/930b36ed48399654c7e1d0e1383e/evaluating-open-qa-evaluation.pdf)
- [LLMs-as-Judges in Free-Form QA (WiNLP 2025)](https://aclanthology.org/2025.winlp-main.37.pdf)
- [SAGE / LLM-as-a-Judge Methodology](https://www.emergentmind.com/topics/llm-as-a-judge-methodology)
