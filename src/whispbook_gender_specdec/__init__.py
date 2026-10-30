"""Offline speaker-gender inference prototype for Whispbook."""

from whispbook_gender_specdec.checkpoints import discover_local_checkpoints
from whispbook_gender_specdec.gender import (
    GenderInferenceResult,
    build_gender_prompt,
    parse_gender_response,
)
from whispbook_gender_specdec.speculative import (
    CheckpointSpec,
    DraftSelection,
    SpeculativeStats,
    choose_draft_checkpoint,
)

__all__ = [
    "CheckpointSpec",
    "DraftSelection",
    "GenderInferenceResult",
    "SpeculativeStats",
    "build_gender_prompt",
    "choose_draft_checkpoint",
    "discover_local_checkpoints",
    "parse_gender_response",
]
