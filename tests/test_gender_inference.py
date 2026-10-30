import json

from whispbook_gender_specdec.gender import (
    GenderInferenceResult,
    build_gender_prompt,
    parse_gender_response,
)
from whispbook_gender_specdec.speculative import CheckpointSpec, choose_draft_checkpoint


def test_prompt_requests_minimal_gender_json() -> None:
    prompt = build_gender_prompt(
        "I promised I would return before sunrise, said Maria.",
        source_hint="chapter_01:segment_003",
    )

    assert "male" in prompt
    assert "female" in prompt
    assert "unknown" in prompt
    assert "JSON" in prompt
    assert "chapter_01:segment_003" in prompt


def test_parse_gender_response_accepts_json_inside_model_text() -> None:
    parsed = parse_gender_response(
        'Sure.\n{"gender": "female", "confidence": 0.86, "evidence": "said Maria"}'
    )

    assert parsed == GenderInferenceResult(
        gender="female",
        confidence=0.86,
        evidence="said Maria",
        raw_response='Sure.\n{"gender": "female", "confidence": 0.86, "evidence": "said Maria"}',
    )


def test_parse_gender_response_rejects_invalid_gender() -> None:
    parsed = parse_gender_response(json.dumps({"gender": "robot", "confidence": 0.2}))

    assert parsed.gender == "unknown"
    assert parsed.confidence == 0.2


def test_choose_draft_checkpoint_prefers_compatible_smollm_then_self_draft() -> None:
    target = CheckpointSpec(
        family="qwen",
        label="Qwen",
        path="/tmp/qwen",
        available=True,
        vocab_size=151936,
        notes=(),
    )
    compatible_smollm = CheckpointSpec(
        family="smollm",
        label="SmolLM",
        path="/tmp/smollm",
        available=True,
        vocab_size=151936,
        notes=(),
    )
    incompatible_gemma = CheckpointSpec(
        family="gemma",
        label="Gemma",
        path="/tmp/gemma",
        available=True,
        vocab_size=262144,
        notes=(),
    )

    chosen = choose_draft_checkpoint(target, [incompatible_gemma, compatible_smollm])
    assert chosen.spec.family == "smollm"
    assert chosen.mode == "external_draft"

    fallback = choose_draft_checkpoint(target, [incompatible_gemma])
    assert fallback.spec.family == "qwen"
    assert fallback.mode == "self_draft"
