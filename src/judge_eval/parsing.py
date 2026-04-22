from __future__ import annotations

import json
import re
from typing import Any


LABEL_RE = re.compile(r'"label"\s*:\s*(true|false)', re.IGNORECASE)
BOOL_RE = re.compile(r"\b(true|false|correct|incorrect)\b", re.IGNORECASE)
THOUGHT_BLOCK_RE = re.compile(r"<thought>.*?</thought>", re.IGNORECASE | re.DOTALL)
CODE_BLOCK_RE = re.compile(r"```[^\n]*\n(.*?)\n```", re.DOTALL)
JSON_OBJECT_RE = re.compile(r"(\{.*?\"label\"\s*:\s*(?:true|false).*?\})", re.IGNORECASE | re.DOTALL)


def _json_parse_candidates(text: str) -> list[str]:
    stripped = text.strip()
    without_thought = THOUGHT_BLOCK_RE.sub("", stripped).strip()
    candidates: list[str] = []
    for candidate in (stripped, without_thought):
        if candidate and candidate not in candidates:
            candidates.append(candidate)
        for match in CODE_BLOCK_RE.findall(candidate):
            block = match.strip()
            if block and block not in candidates:
                candidates.append(block)
        for match in JSON_OBJECT_RE.findall(candidate):
            obj = match.strip()
            if obj and obj not in candidates:
                candidates.append(obj)
    return candidates


def parse_model_output(raw_output: str) -> dict[str, Any]:
    text = raw_output.strip()
    normalized_text = THOUGHT_BLOCK_RE.sub("", text).strip()
    for candidate in _json_parse_candidates(text):
        try:
            payload = json.loads(candidate)
            label = payload.get("label")
            if isinstance(label, bool):
                reason = payload.get("reason")
                return {"parse_method": "json", "parsed_label": label, "judge_reason": reason}
        except json.JSONDecodeError:
            continue
    label_match = LABEL_RE.search(normalized_text)
    if label_match:
        return {
            "parse_method": "regex",
            "parsed_label": label_match.group(1).lower() == "true",
            "judge_reason": None,
        }
    bool_match = BOOL_RE.search(normalized_text)
    if bool_match:
        token = bool_match.group(1).lower()
        return {
            "parse_method": "regex",
            "parsed_label": token in {"true", "correct"},
            "judge_reason": None,
        }
    return {"parse_method": "invalid", "parsed_label": None, "judge_reason": None}
