from __future__ import annotations

import json
import re
from typing import Any


LABEL_RE = re.compile(r'"label"\s*:\s*(true|false)', re.IGNORECASE)
BOOL_RE = re.compile(r"\b(true|false|correct|incorrect)\b", re.IGNORECASE)


def parse_model_output(raw_output: str) -> dict[str, Any]:
    text = raw_output.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[^\n]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text.rstrip())
        text = text.strip()
    try:
        payload = json.loads(text)
        label = payload.get("label")
        if isinstance(label, bool):
            reason = payload.get("reason")
            return {"parse_method": "json", "parsed_label": label, "judge_reason": reason}
    except json.JSONDecodeError:
        pass
    label_match = LABEL_RE.search(text)
    if label_match:
        return {
            "parse_method": "regex",
            "parsed_label": label_match.group(1).lower() == "true",
            "judge_reason": None,
        }
    bool_match = BOOL_RE.search(text)
    if bool_match:
        token = bool_match.group(1).lower()
        return {
            "parse_method": "regex",
            "parsed_label": token in {"true", "correct"},
            "judge_reason": None,
        }
    return {"parse_method": "invalid", "parsed_label": None, "judge_reason": None}
