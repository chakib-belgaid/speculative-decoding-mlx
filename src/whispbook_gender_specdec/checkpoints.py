from __future__ import annotations

import json
import os
from pathlib import Path

from whispbook_gender_specdec.speculative import CheckpointSpec


_KNOWN_VOCAB_SIZES = {
    "mlx-community/Qwen3-0.6B-4bit": 151936,
    "mlx-community/Qwen3-1.7B-4bit": 151936,
    "mlx-community/Qwen3-4B-Instruct-2507-4bit": 151936,
    "mlx-community/Llama-3.2-3B-Instruct-4bit": 128256,
    "mlx-community/Mistral-7B-Instruct-v0.3-4bit": 32768,
    "mlx-community/SmolLM3-3B-4bit": 128256,
    "HuggingFaceTB/SmolLM3-3B": 128256,
    "mlx-community/functiongemma-270m-it-4bit": 262144,
    "unsloth/functiongemma-270m-it": 262144,
}

_CANDIDATES: dict[str, tuple[tuple[str, str], ...]] = {
    "smollm": (
        ("SmolLM3 3B MLX 4bit", "mlx-community/SmolLM3-3B-4bit"),
        ("SmolLM2 135M Instruct MLX 4bit", "mlx-community/SmolLM2-135M-Instruct-4bit"),
        ("SmolLM2 360M Instruct MLX 4bit", "mlx-community/SmolLM2-360M-Instruct-4bit"),
        ("SmolLM3 3B HF", "HuggingFaceTB/SmolLM3-3B"),
        ("SmolLM2 135M Instruct HF", "HuggingFaceTB/SmolLM2-135M-Instruct"),
        ("SmolLM2 360M Instruct HF", "HuggingFaceTB/SmolLM2-360M-Instruct"),
    ),
    "gemma": (
        ("FunctionGemma 270M IT MLX 4bit", "mlx-community/functiongemma-270m-it-4bit"),
        ("FunctionGemma 270M IT HF", "unsloth/functiongemma-270m-it"),
    ),
    "qwen": (
        ("Qwen3 4B Instruct MLX 4bit", "mlx-community/Qwen3-4B-Instruct-2507-4bit"),
    ),
    "qwen_small": (
        ("Qwen3 0.6B MLX 4bit", "mlx-community/Qwen3-0.6B-4bit"),
        ("Qwen3 1.7B MLX 4bit", "mlx-community/Qwen3-1.7B-4bit"),
    ),
    "qwen_large": (
        ("Qwen3 4B Instruct MLX 4bit", "mlx-community/Qwen3-4B-Instruct-2507-4bit"),
    ),
    "llama_3_2_3b": (
        ("Llama 3.2 3B Instruct MLX 4bit", "mlx-community/Llama-3.2-3B-Instruct-4bit"),
    ),
    "mistral_7b": (
        ("Mistral 7B Instruct v0.3 MLX 4bit", "mlx-community/Mistral-7B-Instruct-v0.3-4bit"),
    ),
}

TARGET_FAMILIES = tuple(_CANDIDATES.keys())


def discover_local_checkpoints(
    hf_cache: Path | None = None,
) -> dict[str, CheckpointSpec]:
    return discover_model_specs(hf_cache=hf_cache, local_only=True)


def discover_model_specs(
    hf_cache: Path | None = None,
    *,
    local_only: bool = False,
) -> dict[str, CheckpointSpec]:
    cache_dir = hf_cache or _default_hf_cache()
    return {
        family: _discover_family(family, candidates, cache_dir, local_only=local_only)
        for family, candidates in _CANDIDATES.items()
    }


def _default_hf_cache() -> Path:
    if value := os.environ.get("HF_HUB_CACHE"):
        return Path(value).expanduser()
    if value := os.environ.get("HF_HOME"):
        return Path(value).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _discover_family(
    family: str,
    candidates: tuple[tuple[str, str], ...],
    cache_dir: Path,
    *,
    local_only: bool,
) -> CheckpointSpec:
    notes: list[str] = []
    for label, repo_id in candidates:
        snapshot = _snapshot_for_repo(cache_dir, repo_id)
        if snapshot is None:
            notes.append(f"missing cache for {repo_id}")
            if local_only:
                continue
            return CheckpointSpec(
                family=family,
                label=label,
                path=None,
                available=True,
                vocab_size=_KNOWN_VOCAB_SIZES.get(repo_id),
                notes=(f"remote repo fallback for {repo_id}",),
                repo_id=repo_id,
                source="remote",
            )
        if not _has_required_files(snapshot):
            notes.append(f"{repo_id} exists but is not a standalone local checkpoint")
            continue
        return CheckpointSpec(
            family=family,
            label=label,
            path=str(snapshot),
            available=True,
            vocab_size=_read_vocab_size(snapshot),
            notes=(f"resolved from {repo_id}",),
            repo_id=repo_id,
            source="local",
        )

    return CheckpointSpec(
        family=family,
        label=family,
        path=None,
        available=False,
        vocab_size=None,
        notes=tuple(notes) or (f"no {family} candidates configured",),
    )


def _snapshot_for_repo(cache_dir: Path, repo_id: str) -> Path | None:
    repo_cache = cache_dir / f"models--{repo_id.replace('/', '--')}"
    snapshots = repo_cache / "snapshots"
    if not snapshots.exists():
        return None
    candidates = [path for path in snapshots.iterdir() if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _has_required_files(snapshot: Path) -> bool:
    if not (snapshot / "config.json").exists():
        return False
    if not (snapshot / "tokenizer.json").exists():
        return False
    return any(path.name.startswith("model") and path.name.endswith(".safetensors") for path in snapshot.iterdir())


def _read_vocab_size(snapshot: Path) -> int | None:
    try:
        config = json.loads((snapshot / "config.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    value = config.get("vocab_size")
    return value if isinstance(value, int) else None
