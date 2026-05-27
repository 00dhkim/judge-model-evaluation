"""
Judge-model optimization methods experiment harness.

Usage:
    uv run python experiments/optim_methods.py --mode sanity
    uv run python experiments/optim_methods.py --mode screening --n-samples 300
    uv run python experiments/optim_methods.py --mode final --n-samples 1200
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from judge_eval.config import ModelConfig
from judge_eval.metrics import scotts_pi
from judge_eval.parsing import parse_model_output
from judge_eval.providers import call_provider

# ---------------------------------------------------------------------------
# Baseline Solar Pro3 config
# ---------------------------------------------------------------------------
SOLAR_CONFIG = ModelConfig(
    name="solar_pro_3_tuned",
    provider="openai_compatible",
    model="solar-pro3",
    endpoint="https://api.upstage.ai/v1/chat/completions",
    api_key_env="UPSTAGE_API_KEY",
    temperature=0.7,
    top_p=0.95,
    frequency_penalty=1.1,
    presence_penalty=0.0,
    reasoning_effort="high",
    max_tokens=16384,
)

EXAONE_CONFIG = ModelConfig(
    name="k_exaone_236b_a23b_tuned",
    provider="openai_compatible",
    model="LGAI-EXAONE/K-EXAONE-236B-A23B",
    endpoint="https://api.friendli.ai/serverless/v1/chat/completions",
    api_key_env="FRIENDLI_API_KEY",
    temperature=0.7,
    top_p=0.95,
    max_tokens=4096,
    reasoning_budget=2048,
)

JUDGE_CONFIGS = {
    "solar": SOLAR_CONFIG,
    "exaone": EXAONE_CONFIG,
}

GT_PARQUETS = {
    "solar": "outputs/20260523_solar_202605_tuned/normalized_samples.parquet",
    "exaone": "outputs/20260523_exaone_202605_tuned/normalized_samples.parquet",
}

# ---------------------------------------------------------------------------
# Prompt constants
# ---------------------------------------------------------------------------
PROMPT_GUIDELINES = (
    "Judge semantic correctness against the golden answer aliases. "
    "Be strict about factual mismatch, but accept equivalent aliases and obvious paraphrases."
)

PROMPT_EXAMPLES = (
    "Example correct: Golden answer 'David Seville', candidate 'David Seville'.\n"
    "Example incorrect: Golden answer 'David Seville', candidate 'Alvin'."
)

EXTENDED_FEWSHOT_EXAMPLES = """Example 1 (alias match):
  Golden: Mark Twain | Candidate: Samuel Clemens | Label: true
  Reason: Samuel Clemens is the birth name of Mark Twain — they are the same person.

Example 2 (partial entity):
  Golden: Barack Obama | Candidate: Obama | Label: true
  Reason: "Obama" unambiguously refers to Barack Obama in context.

Example 3 (country/region alias):
  Golden: United States | Candidate: America | Label: true
  Reason: "America" is a widely accepted synonym for the United States.

Example 4 (numeric variant):
  Golden: 1969 | Candidate: nineteen sixty-nine | Label: true
  Reason: The candidate expresses the same year in words.

Example 5 (wrong related entity):
  Golden: Mark Twain | Candidate: Charles Dickens | Label: false
  Reason: Charles Dickens is a different author; factual mismatch.

Example 6 (verbose with right entity):
  Golden: Tokyo | Candidate: The capital city is Tokyo, Japan | Label: true
  Reason: The candidate contains the correct entity "Tokyo".

Example 7 (hedged wrong answer):
  Golden: Paris | Candidate: I think it might be London | Label: false
  Reason: The candidate suggests London, which is incorrect.

Example 8 (incomplete answer):
  Golden: Romeo and Juliet | Candidate: Romeo | Label: false
  Reason: The candidate names only one character; the complete title is required."""

COT_SYSTEM_PROMPT = (
    "You are a strict judge. Before producing JSON, write 'Reasoning:' followed by "
    "step-by-step analysis (1. entity in candidate, 2. compare to each acceptable form, "
    "3. paraphrase check, 4. final). "
    'Then emit the JSON {"reason":"<brief>","label":true|false}.'
)

DECOMPOSED_TASK = (
    'Task:\nReturn JSON with fields: candidate_entity (str), golden_entities (list[str]), '
    'normalized_match (bool), paraphrase_match (bool), reason (str), '
    'label (bool — the final correctness verdict).'
)

CONFIDENCE_TASK = (
    'Task:\nReturn only JSON with reason first, label second, and confidence third: '
    '{"reason":"brief explanation","label":true,"confidence":0.9}'
)

STANDARD_TASK = (
    'Task:\nReturn only JSON with reason first and label second: {"reason":"brief explanation","label":true}'
)

# ---------------------------------------------------------------------------
# MethodSpec dataclass
# ---------------------------------------------------------------------------
@dataclass
class MethodSpec:
    name: str
    sc_n: int = 1
    alias_enum: bool = False
    alias_shuffle: bool = False
    explicit_cot: bool = False
    decomposed: bool = False
    extended_fewshot: bool = False
    confidence_abstain: bool = False
    abstain_threshold: float = 0.6


# ---------------------------------------------------------------------------
# Individual methods
# ---------------------------------------------------------------------------
INDIVIDUAL_METHODS: list[MethodSpec] = [
    MethodSpec("m1_sc_n5", sc_n=5),
    MethodSpec("m2_alias_enum", alias_enum=True),
    MethodSpec("m3_alias_shuffle", alias_enum=True, alias_shuffle=True),
    MethodSpec("m4_explicit_cot", explicit_cot=True),
    MethodSpec("m5_decomposed", decomposed=True),
    MethodSpec("m6_extended_fewshot", extended_fewshot=True),
    MethodSpec("m7_confidence_abstain", confidence_abstain=True),
]

# ---------------------------------------------------------------------------
# Combination methods
# ---------------------------------------------------------------------------
COMBO_METHODS: list[MethodSpec] = [
    # --- Round 1 (pre-individual-method analysis) ---
    MethodSpec("c01_sc5_aliasenum",            sc_n=5, alias_enum=True),
    MethodSpec("c02_sc5_extfewshot",           sc_n=5, extended_fewshot=True),
    MethodSpec("c03_aliasenum_extfewshot",     alias_enum=True, extended_fewshot=True),
    MethodSpec("c04_sc5_aliasenum_extfewshot", sc_n=5, alias_enum=True, extended_fewshot=True),
    MethodSpec("c05_decomposed_extfewshot",    decomposed=True, extended_fewshot=True),
    MethodSpec("c07_sc5_aliasshuffle_extfewshot", sc_n=5, alias_enum=True, alias_shuffle=True, extended_fewshot=True),
    MethodSpec("c10_all_lite", sc_n=3, alias_enum=True, alias_shuffle=False, extended_fewshot=True, confidence_abstain=True),
    # --- Round 2 (m7-confidence_abstain focused combos) ---
    MethodSpec("n01_sc5_confidence",           sc_n=5, confidence_abstain=True),
    MethodSpec("n02_shuffle_confidence",       alias_enum=True, alias_shuffle=True, confidence_abstain=True),
    MethodSpec("n03_sc5_shuffle_confidence",   sc_n=5, alias_enum=True, alias_shuffle=True, confidence_abstain=True),
    MethodSpec("n04_sc5_shuffle_ext_confidence", sc_n=5, alias_enum=True, alias_shuffle=True, extended_fewshot=True, confidence_abstain=True),
    MethodSpec("n05_sc3_shuffle_confidence",   sc_n=3, alias_enum=True, alias_shuffle=True, confidence_abstain=True),
    MethodSpec("n06_sc5_ext_confidence",       sc_n=5, extended_fewshot=True, confidence_abstain=True),
    MethodSpec("n07_shuffle_ext_confidence",   alias_enum=True, alias_shuffle=True, extended_fewshot=True, confidence_abstain=True),
]

ALL_METHODS: list[MethodSpec] = INDIVIDUAL_METHODS + COMBO_METHODS

# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _alias_enum_block(aliases: list[str]) -> str:
    lines = "\n".join(f"- {a}" for a in aliases if a.strip())
    return (
        f"Acceptable answer forms (any equivalent):\n{lines}\n"
        "The candidate is correct if it matches ANY of these (or any obvious paraphrase)."
    )


def build_prompt_for_method(
    sample: dict,
    spec: MethodSpec,
    alias_order: list[str] | None = None,
) -> str:
    question = str(sample["question"])
    golden_answer = str(sample["golden_answer"])
    raw_aliases = sample.get("golden_aliases", [])
    if isinstance(raw_aliases, str):
        try:
            raw_aliases = json.loads(raw_aliases)
        except json.JSONDecodeError:
            raw_aliases = [raw_aliases]
    aliases: list[str] = [str(a) for a in raw_aliases]
    candidate_answer = str(sample["candidate_answer"])

    if alias_order is not None:
        aliases = alias_order

    parts = [
        f"Question:\n{question}",
        f"Golden answer:\n{golden_answer}",
        f"Golden answer aliases:\n{json.dumps(aliases, ensure_ascii=False)}",
    ]

    if spec.alias_enum:
        parts.append(_alias_enum_block(aliases))

    parts.append(f"Candidate answer:\n{candidate_answer}")
    parts.append(f"Guidelines:\n{PROMPT_GUIDELINES}")

    examples = EXTENDED_FEWSHOT_EXAMPLES if spec.extended_fewshot else PROMPT_EXAMPLES
    parts.append(f"Examples:\n{examples}")

    if spec.decomposed:
        parts.append(DECOMPOSED_TASK)
    elif spec.confidence_abstain:
        parts.append(CONFIDENCE_TASK)
    else:
        parts.append(STANDARD_TASK)

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

CONFIDENCE_RE = re.compile(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)')


def _parse_confidence(raw_output: str) -> float | None:
    m = CONFIDENCE_RE.search(raw_output)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def lexical_match(aliases: list[str], candidate: str) -> bool:
    c = candidate.lower()
    return any(a.lower() in c or c in a.lower() for a in aliases if a.strip())


def aggregate(
    parsed_labels: list[bool | None],
    confidences: list[float | None] | None = None,
) -> bool | None:
    valid = [label for label in parsed_labels if label is not None]
    if not valid:
        return None
    true_count = sum(1 for label in valid if label)
    false_count = len(valid) - true_count
    if true_count >= false_count:
        return True
    return False


# ---------------------------------------------------------------------------
# Sample-level execution
# ---------------------------------------------------------------------------

def run_sample(spec: MethodSpec, sample: dict, model: ModelConfig) -> dict:
    """Execute a single sample under the given MethodSpec. Returns result dict."""
    raw_aliases = sample.get("golden_aliases", [])
    if isinstance(raw_aliases, str):
        try:
            raw_aliases = json.loads(raw_aliases)
        except json.JSONDecodeError:
            raw_aliases = [raw_aliases]
    aliases: list[str] = [str(a) for a in raw_aliases]

    total_latency_ms = 0
    total_cost = 0.0
    n_calls = 0

    # Build effective model (system_prompt for explicit_cot)
    effective_model = model
    if spec.explicit_cot:
        effective_model = model.model_copy(update={"system_prompt": COT_SYSTEM_PROMPT})

    # --- alias_shuffle: 3-call protocol ---
    if spec.alias_shuffle:
        rng_a = random.Random(0)
        rng_b = random.Random(1)
        order_a = rng_a.sample(aliases, len(aliases)) if aliases else aliases
        order_b = rng_b.sample(aliases, len(aliases)) if aliases else aliases

        prompt_a = build_prompt_for_method(sample, spec, alias_order=order_a)
        prompt_b = build_prompt_for_method(sample, spec, alias_order=order_b)

        with ThreadPoolExecutor(max_workers=2) as ab_exec:
            fut_a = ab_exec.submit(call_provider, effective_model, prompt_a)
            fut_b = ab_exec.submit(call_provider, effective_model, prompt_b)
            resp_a = fut_a.result()
            resp_b = fut_b.result()

        total_latency_ms += resp_a.latency_ms + resp_b.latency_ms
        total_cost += (resp_a.estimated_cost or 0.0) + (resp_b.estimated_cost or 0.0)
        n_calls += 2
        label_a = parse_model_output(resp_a.raw_output)["parsed_label"]
        label_b = parse_model_output(resp_b.raw_output)["parsed_label"]

        if label_a == label_b and label_a is not None:
            final_label = label_a
        else:
            # tiebreak: 3rd call with original alias order
            prompt_c = build_prompt_for_method(sample, spec, alias_order=None)
            resp_c = call_provider(effective_model, prompt_c)
            total_latency_ms += resp_c.latency_ms
            total_cost += resp_c.estimated_cost or 0.0
            n_calls += 1
            label_c = parse_model_output(resp_c.raw_output)["parsed_label"]
            final_label = aggregate([label_a, label_b, label_c])

        return {
            "sample_id": sample["sample_id"],
            "method": spec.name,
            "parsed_label": final_label,
            "n_calls": n_calls,
            "total_latency_ms": total_latency_ms,
            "total_cost": total_cost,
        }

    # --- sc_n > 1: self-consistency / majority vote ---
    if spec.sc_n > 1:
        prompt = build_prompt_for_method(sample, spec)
        labels: list[bool | None] = []
        confidences: list[float | None] = []

        def _sc_call(_: int):
            r = call_provider(effective_model, prompt)
            return r, parse_model_output(r.raw_output)["parsed_label"], (
                _parse_confidence(r.raw_output) if spec.confidence_abstain else None
            )

        with ThreadPoolExecutor(max_workers=spec.sc_n) as sc_exec:
            sc_results = list(sc_exec.map(_sc_call, range(spec.sc_n)))

        for r, lbl, conf in sc_results:
            total_latency_ms += r.latency_ms
            total_cost += r.estimated_cost or 0.0
            n_calls += 1
            labels.append(lbl)
            if spec.confidence_abstain:
                confidences.append(conf)

        if spec.confidence_abstain:
            # apply per-call abstain logic before aggregating
            adjusted: list[bool | None] = []
            for lbl, conf in zip(labels, confidences):
                if conf is not None and conf < spec.abstain_threshold:
                    adjusted.append(lexical_match(aliases, str(sample["candidate_answer"])))
                elif lbl is None:
                    adjusted.append(lexical_match(aliases, str(sample["candidate_answer"])))
                else:
                    adjusted.append(lbl)
            final_label = aggregate(adjusted)
        else:
            final_label = aggregate(labels)

        return {
            "sample_id": sample["sample_id"],
            "method": spec.name,
            "parsed_label": final_label,
            "n_calls": n_calls,
            "total_latency_ms": total_latency_ms,
            "total_cost": total_cost,
        }

    # --- single call ---
    prompt = build_prompt_for_method(sample, spec)
    resp = call_provider(effective_model, prompt)
    total_latency_ms += resp.latency_ms
    total_cost += resp.estimated_cost or 0.0
    n_calls += 1
    parsed = parse_model_output(resp.raw_output)
    label = parsed["parsed_label"]

    if spec.confidence_abstain:
        conf = _parse_confidence(resp.raw_output)
        if conf is None or conf < spec.abstain_threshold or label is None:
            label = lexical_match(aliases, str(sample["candidate_answer"]))

    return {
        "sample_id": sample["sample_id"],
        "method": spec.name,
        "parsed_label": label,
        "n_calls": n_calls,
        "total_latency_ms": total_latency_ms,
        "total_cost": total_cost,
    }


# ---------------------------------------------------------------------------
# Method-level orchestration
# ---------------------------------------------------------------------------

def run_method(
    spec: MethodSpec,
    samples: pd.DataFrame,
    model: ModelConfig,
    workers: int = 5,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Run a MethodSpec over all samples using ThreadPoolExecutor. Returns DataFrame."""
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    if output_dir is None:
        output_dir = Path("outputs/optim_screening")
    method_dir = output_dir / spec.name
    method_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = method_dir / "predictions.jsonl"

    # Resume: load already-completed sample_ids from existing JSONL
    done_ids: set[str] = set()
    existing_results: list[dict] = []
    if jsonl_path.exists():
        with jsonl_path.open(encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    done_ids.add(str(row["sample_id"]))
                    existing_results.append(row)
        if done_ids:
            print(f"  [resume] {len(done_ids)} samples already done, skipping", file=sys.stderr)

    records = [r for r in samples.to_dict(orient="records") if str(r["sample_id"]) not in done_ids]
    results: list[dict] = list(existing_results)

    def _worker(sample: dict) -> dict:
        return run_sample(spec, sample, model)

    import threading
    _write_lock = threading.Lock()

    with jsonl_path.open("a", encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_worker, rec): rec for rec in records}
            if use_tqdm:
                iterator = tqdm(as_completed(futures), total=len(futures), desc=spec.name, file=sys.stderr)
            else:
                print(f"Running {spec.name} ({len(records)} samples)...", file=sys.stderr)
                iterator = as_completed(futures)

            for future in iterator:
                try:
                    result = future.result(timeout=240)
                except Exception as exc:
                    # Timed-out or errored worker: skip sample, log to stderr
                    sample_rec = futures[future]
                    print(f"\n[WARN] sample {sample_rec.get('sample_id')} failed/timed-out: {exc}", file=sys.stderr)
                    continue
                with _write_lock:
                    fout.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
                    fout.flush()
                results.append(result)

    df = pd.DataFrame(results)
    # Merge human_label from samples
    label_map = samples.set_index("sample_id")["human_label"].to_dict()
    df["human_label"] = df["sample_id"].map(label_map)
    df.rename(columns={"total_latency_ms": "latency_ms", "total_cost": "cost"}, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate(df: pd.DataFrame) -> dict:
    total = len(df)
    valid = df[df["parsed_label"].notna()].copy()
    coverage = len(valid) / total if total else 0.0

    if valid.empty:
        return {
            "scotts_pi": 0.0,
            "accuracy": 0.0,
            "coverage": coverage,
            "total_calls": int(df["n_calls"].sum()) if "n_calls" in df.columns else 0,
            "total_cost": float(df["cost"].sum()) if "cost" in df.columns else 0.0,
            "avg_latency_ms": 0.0,
        }

    pred = valid["parsed_label"].astype(bool).tolist()
    gold = valid["human_label"].astype(bool).tolist()
    pi = scotts_pi(pred, gold)
    accuracy = float(sum(p == g for p, g in zip(pred, gold)) / len(pred))

    return {
        "scotts_pi": pi,
        "accuracy": accuracy,
        "coverage": coverage,
        "total_calls": int(df["n_calls"].sum()) if "n_calls" in df.columns else 0,
        "total_cost": float(df["cost"].sum()) if "cost" in df.columns else 0.0,
        "avg_latency_ms": float(df["latency_ms"].mean()) if "latency_ms" in df.columns else 0.0,
    }


# ---------------------------------------------------------------------------
# Sample loading
# ---------------------------------------------------------------------------

def load_samples(n: int, seed: int = 42, parquet_path: str | None = None) -> pd.DataFrame:
    if parquet_path is None:
        parquet_path = GT_PARQUETS["solar"]
    df = pd.read_parquet(parquet_path)

    # Stratified sample by dataset (TQ/NQ 50/50)
    datasets = df["dataset"].unique().tolist()
    per_dataset = n // len(datasets)
    remainder = n % len(datasets)

    parts = []
    for i, ds in enumerate(sorted(datasets)):
        subset = df[df["dataset"] == ds]
        k = per_dataset + (1 if i < remainder else 0)
        k = min(k, len(subset))
        parts.append(subset.sample(n=k, random_state=seed))

    result = pd.concat(parts, ignore_index=True)
    return result.sample(frac=1, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Judge optimization method screening")
    parser.add_argument("--mode", choices=["sanity", "screening", "final"], required=True)
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--judge", choices=["solar", "exaone"], default="solar")
    parser.add_argument(
        "--methods",
        nargs="*",
        help="Subset of method names to run (default: all individual + combo)",
    )
    args = parser.parse_args()

    judge_config = JUDGE_CONFIGS[args.judge]
    gt_parquet = GT_PARQUETS[args.judge]
    judge_tag = "" if args.judge == "solar" else f"_{args.judge}"

    if args.mode == "sanity":
        n_samples = 3
        method_names = ["m1_sc_n5", "m2_alias_enum"]
        output_dir = Path(f"outputs/optim_screening{judge_tag}")
    elif args.mode == "screening":
        n_samples = args.n_samples
        method_names = args.methods or [s.name for s in ALL_METHODS]
        output_dir = Path(f"outputs/optim_screening{judge_tag}")
    else:  # final
        n_samples = args.n_samples
        method_names = args.methods or [s.name for s in ALL_METHODS]
        output_dir = Path(f"outputs/optim_final{judge_tag}")

    output_dir.mkdir(parents=True, exist_ok=True)

    name_to_spec = {s.name: s for s in ALL_METHODS}
    specs_to_run: list[MethodSpec] = []
    for name in method_names:
        if name not in name_to_spec:
            print(f"Warning: unknown method '{name}', skipping", file=sys.stderr)
            continue
        specs_to_run.append(name_to_spec[name])

    if not specs_to_run:
        print("No valid methods to run.", file=sys.stderr)
        sys.exit(1)

    print(f"Mode: {args.mode} | Judge: {args.judge} | Samples: {n_samples} | Methods: {[s.name for s in specs_to_run]}")
    samples = load_samples(n_samples, parquet_path=gt_parquet)
    print(f"Loaded {len(samples)} samples from {samples['dataset'].value_counts().to_dict()}")

    summary_rows: list[dict] = []
    for spec in specs_to_run:
        print(f"\n--- Running {spec.name} ---")
        df = run_method(spec, samples, judge_config, workers=args.workers, output_dir=output_dir)
        metrics = evaluate(df)
        row = {"method": spec.name, **metrics}
        summary_rows.append(row)
        print(
            f"  pi={metrics['scotts_pi']:.4f}  acc={metrics['accuracy']:.4f}  "
            f"cov={metrics['coverage']:.3f}  calls={metrics['total_calls']}  "
            f"cost=${metrics['total_cost']:.4f}  latency={metrics['avg_latency_ms']:.0f}ms"
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("scotts_pi", ascending=False)

    suffix = "_final" if args.mode == "final" else ""
    summary_path = output_dir / f"summary{suffix}.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 72)
    print("SUMMARY (sorted by scotts_pi DESC)")
    print("=" * 72)
    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
            columns=["method", "scotts_pi", "accuracy", "coverage", "total_calls", "total_cost", "avg_latency_ms"],
        )
    )
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
