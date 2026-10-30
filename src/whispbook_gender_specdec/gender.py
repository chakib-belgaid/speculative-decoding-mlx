from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from whispbook_gender_specdec.speculative import SpeculativeStats


ALLOWED_GENDERS = {"male", "female", "unknown"}


@dataclass(frozen=True)
class GenderInferenceResult:
    gender: str
    confidence: float
    evidence: str
    raw_response: str
    speculative: SpeculativeStats | None = None


def build_gender_prompt(text: str, source_hint: str | None = None) -> str:
    source = source_hint or "unknown"
    return f"""No tools. Infer the likely gender of the speaker in this text for audiobook voice routing.

Use only explicit textual evidence such as names, titles, pronouns, or speaker tags. If the text does not provide enough evidence, use "unknown".

Return only compact JSON with this exact shape:
{{"gender":"male|female|unknown","confidence":0.0,"evidence":"short quote or reason"}}

Source: {source}
Text:
\"\"\"{text.strip()}\"\"\""""


def parse_gender_response(response: str) -> GenderInferenceResult:
    payload = _extract_first_json_object(response)
    if not isinstance(payload, dict):
        return GenderInferenceResult(
            gender="unknown",
            confidence=0.0,
            evidence="no JSON object found",
            raw_response=response,
        )

    gender = str(payload.get("gender", "unknown")).strip().lower()
    if gender not in ALLOWED_GENDERS:
        gender = "unknown"

    confidence = _coerce_confidence(payload.get("confidence", 0.0))
    evidence = payload.get("evidence", "")
    if not isinstance(evidence, str):
        evidence = json.dumps(evidence, ensure_ascii=True)

    return GenderInferenceResult(
        gender=gender,
        confidence=confidence,
        evidence=evidence.strip(),
        raw_response=response,
    )


def _extract_first_json_object(text: str) -> Any:
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        return payload
    return None


def _coerce_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(max(confidence, 0.0), 1.0)
