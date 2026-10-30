from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal


@dataclass(frozen=True)
class CheckpointSpec:
    family: str
    label: str
    path: str | None
    available: bool
    vocab_size: int | None
    notes: tuple[str, ...] = ()
    repo_id: str | None = None
    source: str = "local"

    @property
    def model_ref(self) -> str | None:
        return self.path or self.repo_id


@dataclass(frozen=True)
class DraftSelection:
    spec: CheckpointSpec
    mode: Literal["external_draft", "self_draft", "unavailable"]
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class SpeculativeStats:
    target_family: str
    draft_family: str
    draft_mode: str
    total_tokens: int
    draft_tokens: int
    finish_reason: str

    @property
    def draft_acceptance_ratio(self) -> float:
        if self.total_tokens == 0:
            return 0.0
        return self.draft_tokens / self.total_tokens


def choose_draft_checkpoint(
    target: CheckpointSpec,
    candidates: Iterable[CheckpointSpec],
) -> DraftSelection:
    if not target.available or target.model_ref is None:
        return DraftSelection(
            spec=target,
            mode="unavailable",
            notes=("target checkpoint is not available",),
        )

    incompatible: list[str] = []
    for candidate in candidates:
        if candidate.family == target.family:
            continue
        if not candidate.available or candidate.model_ref is None:
            continue
        if candidate.vocab_size == target.vocab_size:
            return DraftSelection(
                spec=candidate,
                mode="external_draft",
                notes=(
                    f"{candidate.label} shares vocab_size={target.vocab_size} with {target.label}",
                ),
            )
        incompatible.append(
            f"{candidate.family}: vocab_size={candidate.vocab_size} != {target.vocab_size}"
        )

    notes = (
        "no compatible external draft checkpoint found; using target as its own draft",
        *incompatible,
    )
    return DraftSelection(spec=target, mode="self_draft", notes=notes)
