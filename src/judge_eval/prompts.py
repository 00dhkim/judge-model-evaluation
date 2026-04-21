from __future__ import annotations

import json


PROMPT_GUIDELINES = (
    "Judge semantic correctness against the golden answer aliases. "
    "Be strict about factual mismatch, but accept equivalent aliases and obvious paraphrases."
)

PROMPT_EXAMPLES = (
    "Example correct: Golden answer 'David Seville', candidate 'David Seville'.\n"
    "Example incorrect: Golden answer 'David Seville', candidate 'Alvin'."
)


def build_prompt(
    question: str,
    golden_answer: str,
    golden_aliases: list[str],
    candidate_answer: str,
    template: str,
) -> str:
    aliases = [str(item) for item in list(golden_aliases)]
    parts = [
        f"Question:\n{question}",
        f"Golden answer:\n{golden_answer}",
        f"Golden answer aliases:\n{json.dumps(aliases, ensure_ascii=False)}",
        f"Candidate answer:\n{candidate_answer}",
    ]
    if template in {"guideline", "guideline_with_examples"}:
        parts.append(f"Guidelines:\n{PROMPT_GUIDELINES}")
    if template == "guideline_with_examples":
        parts.append(f"Examples:\n{PROMPT_EXAMPLES}")
    parts.append(
        'Task:\nReturn only JSON with reason first and label second: {"reason":"brief explanation","label":true}'
    )
    return "\n\n".join(parts)
