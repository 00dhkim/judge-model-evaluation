"""Project-wide constants."""

from __future__ import annotations

ANSWER_SOURCES = ("fid", "gpt35", "chatgpt", "gpt4", "newbing")
PROMPT_TEMPLATES = ("minimal", "guideline", "guideline_with_examples")
VARIANT_TYPES = ("base", "prompt_sensitivity", "reference_order", "dummy_answer")
INVALID_PARSE_STATUSES = {"invalid", "error"}
ARIZE_ENV_VARS = ("ARIZE_API_KEY", "ARIZE_SPACE_ID")
