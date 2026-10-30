from __future__ import annotations

from dataclasses import replace

from whispbook_gender_specdec.gender import (
    GenderInferenceResult,
    build_gender_prompt,
    parse_gender_response,
)
from whispbook_gender_specdec.speculative import CheckpointSpec, SpeculativeStats


def infer_with_mlx_speculative(
    *,
    target: CheckpointSpec,
    draft: CheckpointSpec,
    text: str,
    source_hint: str | None = None,
    max_tokens: int = 96,
    num_draft_tokens: int = 3,
) -> GenderInferenceResult:
    return infer_with_mlx(
        target=target,
        draft=draft,
        text=text,
        source_hint=source_hint,
        max_tokens=max_tokens,
        num_draft_tokens=num_draft_tokens,
    )


def infer_with_mlx(
    *,
    target: CheckpointSpec,
    draft: CheckpointSpec | None,
    text: str,
    source_hint: str | None = None,
    max_tokens: int = 96,
    num_draft_tokens: int = 3,
) -> GenderInferenceResult:
    _require_checkpoint(target, "target")
    if draft is not None:
        _require_checkpoint(draft, "draft")

    try:
        from mlx_lm import load, stream_generate
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "mlx-lm is required for live inference. Run with "
            "`uv run --offline --python 3.12 --with mlx-lm==0.31.3 ...`."
        ) from exc

    target_model, tokenizer = load(target.model_ref)
    if draft is None:
        draft_model = None
        draft_family = "none"
        draft_mode = "none"
    elif draft.model_ref == target.model_ref:
        draft_model = target_model
        draft_family = draft.family
        draft_mode = "self_draft"
    else:
        draft_model, draft_tokenizer = load(draft.model_ref)
        if getattr(draft_tokenizer, "vocab_size", None) != getattr(tokenizer, "vocab_size", None):
            raise ValueError("speculative decoding requires matching tokenizer vocab sizes")
        draft_family = draft.family
        draft_mode = "external_draft"

    prompt = _chat_prompt(tokenizer, build_gender_prompt(text, source_hint=source_hint))
    kwargs = {"max_tokens": max_tokens}
    if draft_model is not None:
        kwargs["draft_model"] = draft_model
        kwargs["num_draft_tokens"] = num_draft_tokens
    chunks = list(stream_generate(target_model, tokenizer, prompt, **kwargs))
    response_text = "".join(chunk.text for chunk in chunks)
    parsed = parse_gender_response(response_text)
    stats = _stats_from_chunks(
        chunks=chunks,
        target=target,
        draft_family=draft_family,
        draft_mode=draft_mode,
    )
    return replace(parsed, speculative=stats)


def _chat_prompt(tokenizer: object, user_prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You produce terse JSON for downstream software. Do not include prose.",
        },
        {"role": "user", "content": user_prompt},
    ]
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if apply_chat_template is None:
        return user_prompt
    try:
        return apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _stats_from_chunks(
    *,
    chunks: list[object],
    target: CheckpointSpec,
    draft_family: str,
    draft_mode: str,
) -> SpeculativeStats:
    if not chunks:
        return SpeculativeStats(
            target_family=target.family,
            draft_family=draft_family,
            draft_mode=draft_mode,
            total_tokens=0,
            draft_tokens=0,
            finish_reason="empty",
        )
    total_tokens = max(int(getattr(chunk, "generation_tokens", 0)) for chunk in chunks)
    draft_tokens = sum(1 for chunk in chunks if bool(getattr(chunk, "from_draft", False)))
    finish_reason = str(getattr(chunks[-1], "finish_reason", None) or "unknown")
    return SpeculativeStats(
        target_family=target.family,
        draft_family=draft_family,
        draft_mode=draft_mode,
        total_tokens=total_tokens,
        draft_tokens=draft_tokens,
        finish_reason=finish_reason,
    )


def _require_checkpoint(spec: CheckpointSpec, role: str) -> None:
    if not spec.available or spec.model_ref is None:
        raise FileNotFoundError(f"{role} checkpoint is unavailable: {spec.notes}")
