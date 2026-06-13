# Judge Model Optimization Methods

이 문서는 `merged_report_optimized.html`에 반영된 judge model 최적화 방법을 연구자 관점에서 설명한다. 대상 태스크는 EVOUNA의 `question + golden_answer + candidate_answer`를 입력받아 candidate answer의 정오를 `true/false`로 판정하는 LLM-as-a-judge 설정이다. 최적화의 목표는 judge model의 예측 라벨을 EVOUNA human label과 더 잘 일치시키는 것이며, 주 평가지표는 Scott's π다.

프로젝트 구현은 [experiments/optim_methods.py](experiments/optim_methods.py)에 있고, 실험 요약은 [experiments/README.md](experiments/README.md)에 있다. 최종 optimized report에는 Solar Pro3에 `n06_sc5_ext_confidence`, K-EXAONE-236B-A23B에 `c10_all_lite`가 적용되었다.

## 평가 설정

각 샘플은 다음 구조를 가진다.

```text
Question: q
Golden answer: g
Golden answer aliases: A = {a_1, ..., a_m}
Candidate answer: c
Human label: y ∈ {true, false}
```

judge model은 `ŷ ∈ {true, false}`를 출력한다. 최적화 방법은 프롬프트, 반복 호출, alias 처리, confidence 처리 방식을 바꾸어 `ŷ`와 `y`의 일치를 높인다.

주요 평가지표는 다음과 같다.

```text
accuracy = (TP + TN) / (TP + FP + TN + FN)

Scott's π = (P_o - P_e) / (1 - P_e)
```

여기서 `P_o`는 judge label과 human label의 관측 일치율이고, `P_e`는 두 평가자가 라벨 분포만으로 우연히 일치할 기대 확률이다. class imbalance가 있는 데이터에서는 단순 accuracy보다 Scott's π가 judge-human alignment를 더 엄격하게 본다.

## 최종 조합

최종 report에 들어간 두 optimized judge는 다음 조합을 사용한다.

| Optimized judge | Method ID | 구성 |
|---|---|---|
| `solar_pro_3_optimized` | `n06_sc5_ext_confidence` | `m1` Self-Consistency K=5 + `m6` Extended Few-shot + `m7` Confidence Abstain |
| `k_exaone_236b_a23b_optimized` | `c10_all_lite` | `m1` Self-Consistency K=3 + `m2` Alias Enumeration + `m6` Extended Few-shot + `m7` Confidence Abstain |

개별 방법과 조합 목록은 `MethodSpec`으로 정의된다. 구현상 `sc_n`, `alias_enum`, `alias_shuffle`, `extended_fewshot`, `confidence_abstain` 같은 boolean/parameter flag를 조합한다.

## m1: Self-Consistency

관련 논문:

- [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171)
- [Confidence Improves Self-Consistency in LLMs](https://aclanthology.org/2025.findings-acl.1030/)

Self-Consistency는 같은 문제에 대해 모델을 여러 번 샘플링한 뒤 다수결로 최종 답을 정하는 방법이다. 원래는 chain-of-thought reasoning에서 서로 다른 reasoning path를 여러 개 생성하고 가장 자주 나온 답을 선택하는 방식으로 제안되었다. judge setting에서는 reasoning path 대신 `label`을 여러 번 얻고 다수결한다.

이 프로젝트의 구현은 다음과 같다.

```text
for k = 1..K:
    r_k = judge(q, g, A, c)
    y_k = parse_label(r_k)

ŷ = majority_vote({y_1, ..., y_K})
```

동률이 생기면 현재 구현의 `aggregate()`는 `true_count >= false_count`일 때 `true`를 반환한다. 즉 순수 Self-Consistency에서는 동률이 정답 쪽으로 기울 수 있다. `m7`과 결합된 경우에는 각 호출의 confidence를 먼저 반영한 뒤 다수결한다.

수식으로 쓰면 라벨 집합 `Y = {y_1, ..., y_K}`에 대해:

```text
ŷ = argmax_{b ∈ {true,false}} Σ_k 1[y_k = b]
```

Confidence-Informed Self-Consistency(CISC)는 여기에 confidence weight를 도입한다.

```text
ŷ = argmax_{b ∈ {true,false}} Σ_k w_k · 1[y_k = b]
```

이 프로젝트는 완전한 confidence-weighted vote보다는 `confidence_abstain` 전처리 후 majority vote에 가깝다. 그래도 핵심 아이디어는 같다. 단일 호출의 우발적 오판을 줄이고, 반복 호출에서 안정적으로 나오는 판정을 채택한다.

장점:

- judge의 stochastic error를 줄인다.
- 작은 프롬프트 수정 없이 적용할 수 있다.
- `m6`, `m7`과 결합했을 때 효과가 컸다.

비용:

- 호출 수가 K배 증가한다.
- 이 프로젝트 최종 Solar 조합은 K=5, EXAONE 조합은 K=3을 사용한다.

## m2: Alias Enumeration

관련 논문:

- [CLEV: LLM-Based Evaluation Through Lightweight Efficient Voting for Free-Form Question-Answering](https://arxiv.org/abs/2503.08542)
- [LLMs-as-Judges in Automatic Evaluation of Free-Form QA](https://aclanthology.org/2025.winlp-main.37/)

EVOUNA의 `golden_answer`는 `/`로 연결된 여러 alias를 포함한다. 예를 들어 `Moonwalk/Moon Walk/Moonwalking`처럼 같은 정답의 표기 변형이 들어간다. 기본 프롬프트에도 alias list가 들어가지만, `m2_alias_enum`은 이를 더 명시적인 판정 규칙으로 바꾼다.

프로젝트 구현은 프롬프트에 다음 블록을 추가한다.

```text
Acceptable answer forms (any equivalent):
- alias_1
- alias_2
- ...
The candidate is correct if it matches ANY of these (or any obvious paraphrase).
```

목표는 false negative를 줄이는 것이다. 즉 candidate가 golden answer의 표면형과 다르더라도 alias나 명백한 paraphrase에 해당하면 정답으로 판정하게 한다.

판정 관점에서는 다음 조건을 judge에게 강하게 전달한다.

```text
correct(c, A) = true
if ∃ a_i ∈ A such that semantic_equivalent(c, a_i)
```

CLEV와 Reference-Guided Verdict 계열의 핵심은 free-form QA 평가에서 reference answer를 더 적극적으로 사용해 binary verdict를 안정화하는 것이다. 이 프로젝트의 `m2`는 multi-judge voting 자체를 그대로 구현한 것은 아니며, EVOUNA에 이미 존재하는 alias 정보를 reference-guided verdict에 맞게 프롬프트에 명시한 변형이다.

장점:

- short-form QA에서 alias mismatch로 인한 false negative를 줄일 수 있다.
- EVOUNA처럼 alias가 이미 구조적으로 들어있는 데이터셋에 적용하기 쉽다.

주의점:

- 단독 사용은 항상 성능을 올리지 않았다. Solar 스크리닝에서는 `m2_alias_enum` 단독이 baseline보다 낮았다.
- alias 품질이 낮거나 과도하게 넓으면 false positive가 늘 수 있다.
- 최종 EXAONE 조합에서는 `m2`가 `m1`, `m6`, `m7`과 함께 쓰일 때 가장 좋은 결과를 냈다.

## m3: Alias Shuffle

관련 논문:

- [JudgeLM: Fine-tuned Large Language Models are Scalable Judges](https://arxiv.org/abs/2310.17631)

JudgeLM은 LLM-as-a-judge에서 position bias, knowledge bias, format bias를 분석하고, swap augmentation, reference support, reference drop 같은 기법을 제안한다. `m3_alias_shuffle`은 이 중 position/order bias 완화 아이디어를 EVOUNA alias 순서에 적용한 것이다.

EVOUNA의 golden aliases는 리스트 순서를 가진다. 어떤 judge는 앞쪽 alias에 과도하게 의존하거나, 긴 alias 목록에서 뒤쪽 항목을 덜 반영할 수 있다. `m3`는 alias 순서를 두 가지 seed로 섞어 2회 판정하고, 불일치하면 원래 순서로 한 번 더 판정해 다수결한다.

구현 절차:

```text
A_0 = original alias order
A_1 = shuffle(A, seed=0)
A_2 = shuffle(A, seed=1)

y_1 = judge(q, g, A_1, c)
y_2 = judge(q, g, A_2, c)

if y_1 == y_2 and y_1 is valid:
    ŷ = y_1
else:
    y_3 = judge(q, g, A_0, c)
    ŷ = majority_vote(y_1, y_2, y_3)
```

이 방식은 CLEV의 "두 judge가 동의하면 채택, 불일치하면 추가 judge 호출" 구조와도 유사하지만, 여기서는 서로 다른 모델을 쓰는 대신 같은 모델에 alias order perturbation을 준다.

장점:

- alias 순서나 위치에 민감한 judge의 variance를 줄인다.
- 두 번의 판정이 일치하면 2회 호출로 끝나고, 불일치할 때만 3회 호출한다.

주의점:

- alias가 하나뿐인 샘플에서는 효과가 제한적이다.
- `m3` 단독은 Solar 스크리닝에서 좋은 편이었지만, 최종 Solar best 조합에는 들어가지 않았다.
- EXAONE에서는 `n05_sc3_shuffle_confidence`가 강했지만 최종 최고는 `c10_all_lite`였다.

## m6: Extended Few-shot

관련 논문/자료:

- [LLMs-as-Judges: A Comprehensive Survey on LLM-based Evaluation Methods](https://arxiv.org/abs/2412.05579)
- [LLM-as-a-Judge: A Survey](https://arxiv.org/abs/2411.15594)

Few-shot prompting은 judge에게 판정 예시를 제공해 decision boundary를 더 명확히 만드는 방법이다. 이 프로젝트의 기본 프롬프트도 짧은 예시를 포함할 수 있지만, `m6_extended_fewshot`은 Open-QA correctness에서 자주 발생하는 혼동 패턴을 더 많이 보여준다.

예시가 겨냥하는 패턴은 다음과 같다.

- alias match: 표기는 다르지만 같은 entity인 경우
- paraphrase: 의미가 같은 표현
- partial answer: 핵심 정보가 빠진 부분정답
- wrong entity: 질문과 관련 있지만 정답 entity가 다른 경우
- over-broad answer: 너무 넓거나 구체성이 부족한 답
- temporal/context mismatch: 질문 시점이나 문맥과 충돌하는 답

프롬프트 레벨에서는 다음 구조가 된다.

```text
Question
Golden answer
Golden answer aliases
Candidate answer
Guidelines
Examples: extended few-shot examples
Task: return JSON label
```

중요한 제한:

- [experiments/README.md](experiments/README.md)는 `m6`의 extended few-shot 8개가 EVOUNA 실제 평가 행에서 추출한 hard-negative라고 주장하지 않는다고 명시한다.
- 따라서 논문식 "train/dev에서 채굴한 hard negatives"라기보다, 일반적인 QA judge 오류 유형을 수동으로 압축한 instruction/examples block으로 보는 것이 정확하다.

장점:

- judge가 "비슷해 보이지만 오답"인 케이스를 더 엄격하게 볼 수 있다.
- `m1`과 `m7`처럼 variance/calibration을 줄이는 방법과 결합했을 때 효과가 컸다.

주의점:

- 예시가 많아지면 토큰 비용이 증가한다.
- 예시 분포가 실제 데이터와 맞지 않으면 편향을 주입할 수 있다.
- 단독 효과보다 조합 효과가 더 중요했다.

## m7: Confidence Abstain

관련 논문/자료:

- [Confidence Improves Self-Consistency in LLMs](https://aclanthology.org/2025.findings-acl.1030/)
- [LLMs-as-Judges: A Comprehensive Survey on LLM-based Evaluation Methods](https://arxiv.org/abs/2412.05579)

`m7_confidence_abstain`은 judge에게 라벨뿐 아니라 confidence를 함께 요구한다.

```json
{"reason": "brief explanation", "label": true, "confidence": 0.9}
```

일반적인 confidence abstention은 confidence가 threshold보다 낮으면 판단을 보류하거나 별도 처리한다. 이 프로젝트의 구현은 보류 샘플을 버리지 않고, confidence가 낮거나 parsing이 실패한 경우 alias 기반 lexical fallback으로 라벨을 대체한다.

구현상 threshold는 기본 `0.6`이다.

```text
if confidence is None:
    ŷ = lexical_match(A, c)
elif confidence < 0.6:
    ŷ = lexical_match(A, c)
elif parsed_label is None:
    ŷ = lexical_match(A, c)
else:
    ŷ = parsed_label
```

`lexical_match(A, c)`는 다음과 같은 단순 포함 관계를 본다.

```text
lexical_match(A, c) = ∃ a_i ∈ A:
    lower(a_i) is substring of lower(c)
    or lower(c) is substring of lower(a_i)
```

Self-Consistency와 결합된 경우에는 각 call의 low-confidence label을 먼저 lexical fallback으로 보정한 뒤 다수결한다.

```text
for k = 1..K:
    if conf_k < τ or label_k invalid:
        z_k = lexical_match(A, c)
    else:
        z_k = label_k

ŷ = majority_vote({z_1, ..., z_K})
```

Confidence-Informed Self-Consistency 논문은 confidence를 vote weight로 사용하는 방향을 제안한다. 이 프로젝트의 `m7`은 weight를 직접 곱하는 방식은 아니지만, 낮은 confidence 출력을 deterministic lexical rule로 대체해 불안정한 judge 출력을 줄인다.

장점:

- 불확실한 LLM 판정을 그대로 믿지 않는다.
- short-form QA에서는 alias lexical fallback이 강한 baseline 역할을 할 수 있다.
- Solar에서는 개별 방법 중 가장 좋은 성능을 보였다.

주의점:

- lexical fallback은 semantic paraphrase를 놓칠 수 있다.
- alias가 과도하게 넓으면 false positive를 만들 수 있다.
- EXAONE에서는 `m7` 단독이 baseline보다 낮았고, 조합에서만 강했다. confidence calibration은 모델별로 다르게 동작한다.

## 방법 간 관계

각 방법은 서로 다른 오류 원인을 겨냥한다.

| 방법 | 겨냥하는 오류 | 주 효과 | 비용 |
|---|---|---|---|
| `m1` Self-Consistency | stochastic variance | 불안정한 단일 호출 완화 | 호출 수 K배 |
| `m2` Alias Enumeration | alias/paraphrase false negative | reference 명시화 | 토큰 증가 |
| `m3` Alias Shuffle | alias order/position bias | 순서 민감도 완화 | 2-3회 호출 |
| `m6` Extended Few-shot | decision boundary 불명확 | 오답 유형 학습 | 토큰 증가 |
| `m7` Confidence Abstain | low-confidence hallucinated judgment | 불확실 판정 보정 | confidence parsing 필요 |

최종 결과에서 좋은 조합은 단일 trick 하나가 아니라, 서로 다른 실패 모드를 동시에 줄인 조합이었다.

```text
Solar best:
    n06 = m1(sc5) + m6 + m7

EXAONE best:
    c10 = m1(sc3) + m2 + m6 + m7
```

## 재현 명령

스크리닝과 final 실험은 다음 entrypoint를 사용한다.

```bash
uv run python experiments/optim_methods.py --mode screening --n-samples 300 --judge solar
uv run python experiments/optim_methods.py --mode final --n-samples 1200 --judge solar

uv run python experiments/optim_methods.py --mode screening --n-samples 300 --judge exaone
uv run python experiments/optim_methods.py --mode final --n-samples 1200 --judge exaone
```

결과 집계는 다음 스크립트로 전체 `predictions.jsonl`을 다시 스캔해 생성한다.

```bash
uv run python scripts/aggregate_optim_results.py --root outputs/optim_final --judge solar
uv run python scripts/aggregate_optim_results.py --root outputs/optim_final_exaone --judge exaone
```

최종 optimized merged report는 다음 스크립트가 만든다.

```bash
uv run python scripts/build_optimized_merged_report.py
```

## 참고 문헌

- Wang et al., [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171).
- Taubenfeld et al., [Confidence Improves Self-Consistency in LLMs](https://aclanthology.org/2025.findings-acl.1030/).
- Zhu et al., [JudgeLM: Fine-tuned Large Language Models are Scalable Judges](https://arxiv.org/abs/2310.17631).
- Badshah et al., [CLEV: LLM-Based Evaluation Through Lightweight Efficient Voting for Free-Form Question-Answering](https://arxiv.org/abs/2503.08542).
- Badshah and Sajjad, [LLMs-as-Judges in Automatic Evaluation of Free-Form QA](https://aclanthology.org/2025.winlp-main.37/).
- Li et al., [LLMs-as-Judges: A Comprehensive Survey on LLM-based Evaluation Methods](https://arxiv.org/abs/2412.05579).
- Gu et al., [A Survey on LLM-as-a-Judge](https://arxiv.org/abs/2411.15594).
