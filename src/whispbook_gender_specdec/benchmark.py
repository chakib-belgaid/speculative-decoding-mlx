from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

import pandas as pd

from whispbook_gender_specdec.gender import GenderInferenceResult
from whispbook_gender_specdec.speculative import CheckpointSpec


REQUIRED_SAMPLE_COLUMNS = ("sample_id", "speaker", "expected_gender", "text")


@dataclass(frozen=True)
class StorySource:
    title: str
    author: str
    project_gutenberg_id: int
    url: str


@dataclass(frozen=True)
class StorySample:
    sample_id: str
    source: StorySource
    speaker: str
    expected_gender: str
    text: str
    evidence_hint: str


@dataclass(frozen=True)
class ModeConfig:
    name: str
    target: CheckpointSpec
    draft: CheckpointSpec | None


@dataclass(frozen=True)
class BenchmarkPrediction:
    mode: str
    sample_id: str
    speaker: str
    expected_gender: str
    result: GenderInferenceResult
    wall_ms: float

    @property
    def correct(self) -> bool:
        return self.result.gender == self.expected_gender

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["predicted_gender"] = self.result.gender
        payload["correct"] = self.correct
        payload["draft_acceptance_ratio"] = (
            self.result.speculative.draft_acceptance_ratio
            if self.result.speculative is not None
            else 0.0
        )
        return payload


_DOLL_HOUSE = StorySource(
    title="A Doll's House",
    author="Henrik Ibsen",
    project_gutenberg_id=2542,
    url="https://www.gutenberg.org/ebooks/2542",
)


def load_builtin_story_samples() -> list[StorySample]:
    return [
        StorySample("doll_house_nora_001", _DOLL_HOUSE, "NORA", "female", "NORA. You might give me money, Torvald.", "Nora, his wife"),
        StorySample("doll_house_helmer_001", _DOLL_HOUSE, "HELMER", "male", "HELMER. When did my squirrel come home?", "Torvald Helmer"),
        StorySample("doll_house_linde_001", _DOLL_HOUSE, "MRS LINDE", "female", "MRS LINDE. How do you do, Nora?", "Mrs Linde"),
        StorySample("doll_house_rank_001", _DOLL_HOUSE, "DOCTOR RANK", "male", "DOCTOR RANK. Thank you, I shall not disturb you.", "Doctor Rank"),
        StorySample("doll_house_krogstad_001", _DOLL_HOUSE, "KROGSTAD", "male", "KROGSTAD. Excuse me, Mrs Helmer.", "Nils Krogstad"),
        StorySample("doll_house_anne_001", _DOLL_HOUSE, "ANNE", "female", "ANNE. The little ones are begging so hard to be allowed to come in.", "Anne, their nurse"),
        StorySample("doll_house_maid_001", _DOLL_HOUSE, "MAID", "female", "MAID. A lady to see you, ma'am.", "The maid"),
        StorySample("doll_house_porter_001", _DOLL_HOUSE, "PORTER", "male", "PORTER. Sixpence.", "The porter"),
    ]


def story_samples_to_dataframe(samples: Iterable[StorySample]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sample in samples:
        rows.append(
            {
                "sample_id": sample.sample_id,
                "speaker": sample.speaker,
                "expected_gender": sample.expected_gender,
                "text": sample.text,
                "evidence_hint": sample.evidence_hint,
                "source_title": sample.source.title,
                "source_author": sample.source.author,
                "project_gutenberg_id": sample.source.project_gutenberg_id,
                "source_url": sample.source.url,
            }
        )
    return normalize_sample_dataframe(pd.DataFrame(rows))


def load_builtin_story_samples_dataframe() -> pd.DataFrame:
    return story_samples_to_dataframe(load_builtin_story_samples())


def load_ground_truth_dataframe(path: str | Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    source = Path(path)
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}:{line_number} is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{source}:{line_number} must contain a JSON object")
        rows.append(payload)
    return normalize_sample_dataframe(pd.DataFrame(rows))


def load_story_benchmark_dataframe(
    *,
    ground_truth_path: str | Path,
    story_text_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> pd.DataFrame:
    samples = load_ground_truth_dataframe(ground_truth_path)
    if story_text_path is not None:
        story_path = Path(story_text_path)
        samples["story_text_path"] = str(story_path)
        samples.attrs["story_text"] = story_path.read_text(encoding="utf-8")
    if metadata_path is not None:
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("metadata file must contain a JSON object")
        _attach_metadata_columns(samples, metadata)
        samples.attrs["metadata"] = metadata
    return normalize_sample_dataframe(samples)


def normalize_sample_dataframe(samples: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_SAMPLE_COLUMNS if column not in samples.columns]
    if missing:
        raise ValueError(f"sample dataframe is missing required columns: {', '.join(missing)}")

    normalized = samples.copy()
    for column in REQUIRED_SAMPLE_COLUMNS:
        normalized[column] = normalized[column].map(_string_cell)
    normalized["expected_gender"] = normalized["expected_gender"].str.strip().str.lower()

    ordered = [
        *REQUIRED_SAMPLE_COLUMNS,
        *[column for column in normalized.columns if column not in REQUIRED_SAMPLE_COLUMNS],
    ]
    return normalized.loc[:, ordered]


def default_mode_configs(
    manifest: dict[str, CheckpointSpec],
    *,
    small_family: str = "qwen_small",
    large_family: str = "qwen_large",
) -> list[ModeConfig]:
    small = manifest[small_family]
    large = manifest[large_family]
    return [
        ModeConfig("small_only", target=small, draft=None),
        ModeConfig("large_only", target=large, draft=None),
        ModeConfig("speculative", target=large, draft=small),
    ]


InferenceFn = Callable[[ModeConfig, StorySample], GenderInferenceResult]
DataFrameInferenceFn = Callable[[ModeConfig, pd.Series], GenderInferenceResult]
ModelInferenceFn = Callable[..., GenderInferenceResult]


def run_benchmark(
    samples: Iterable[StorySample],
    modes: Iterable[ModeConfig],
    infer: InferenceFn,
) -> list[BenchmarkPrediction]:
    predictions: list[BenchmarkPrediction] = []
    for mode in modes:
        for sample in samples:
            started = time.perf_counter()
            result = infer(mode, sample)
            wall_ms = (time.perf_counter() - started) * 1000
            predictions.append(
                BenchmarkPrediction(
                    mode=mode.name,
                    sample_id=sample.sample_id,
                    speaker=sample.speaker,
                    expected_gender=sample.expected_gender,
                    result=result,
                    wall_ms=wall_ms,
                )
            )
    return predictions


def run_benchmark_dataframe(
    samples: pd.DataFrame,
    modes: Iterable[ModeConfig],
    infer: DataFrameInferenceFn,
    *,
    parameters: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    samples_df = normalize_sample_dataframe(samples)
    rows: list[dict[str, object]] = []
    run_parameters = dict(parameters or {})
    for mode in modes:
        for _, sample in samples_df.iterrows():
            started = time.perf_counter()
            result = infer(mode, sample)
            wall_ms = (time.perf_counter() - started) * 1000
            rows.append(
                _benchmark_result_row(
                    mode=mode,
                    sample=sample,
                    result=result,
                    wall_ms=wall_ms,
                    parameters=run_parameters,
                )
            )
    return pd.DataFrame(rows)


def run_model_benchmark_dataframe(
    *,
    samples: pd.DataFrame,
    manifest: Mapping[str, CheckpointSpec],
    infer_model: ModelInferenceFn,
    small_model_name: str = "qwen_small",
    large_model_name: str = "qwen_large",
    max_tokens: int = 96,
    num_draft_tokens: int = 3,
    extra_parameters: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    model_parameters = {
        "max_tokens": max_tokens,
        "num_draft_tokens": num_draft_tokens,
        **dict(extra_parameters or {}),
    }
    run_parameters = {
        "small_model_name": small_model_name,
        "large_model_name": large_model_name,
        **model_parameters,
    }
    modes = default_mode_configs(
        dict(manifest),
        small_family=small_model_name,
        large_family=large_model_name,
    )

    def infer_row(mode: ModeConfig, sample: pd.Series) -> GenderInferenceResult:
        return infer_model(
            target=mode.target,
            draft=mode.draft,
            text=_string_cell(sample["text"]),
            source_hint=source_hint_from_sample_row(sample),
            **model_parameters,
        )

    return run_benchmark_dataframe(
        samples=samples,
        modes=modes,
        infer=infer_row,
        parameters=run_parameters,
    )


def source_hint_from_sample_row(sample: pd.Series) -> str:
    title = _first_present(sample, "story_title", "source_title")
    sample_id = _string_cell(sample.get("sample_id", ""))
    speaker = _string_cell(sample.get("speaker", ""))
    return ":".join(part for part in [title, sample_id, speaker] if part)


def write_benchmark_csv(results: pd.DataFrame, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    return output


def predictions_to_dataframe(predictions: Iterable[BenchmarkPrediction]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for prediction in predictions:
        result = prediction.result
        speculative = result.speculative
        rows.append(
            {
                "mode": prediction.mode,
                "sample_id": prediction.sample_id,
                "speaker": prediction.speaker,
                "expected_gender": prediction.expected_gender,
                "predicted_gender": result.gender,
                "confidence": result.confidence,
                "evidence": result.evidence,
                "raw_response": result.raw_response,
                "parse_ok": result.raw_response.strip().startswith("{"),
                "correct": prediction.correct,
                "wall_ms": prediction.wall_ms,
                "target_family": speculative.target_family if speculative is not None else "",
                "draft_family": speculative.draft_family if speculative is not None else "",
                "draft_mode": speculative.draft_mode if speculative is not None else "none",
                "total_tokens": speculative.total_tokens if speculative is not None else 0,
                "draft_tokens": speculative.draft_tokens if speculative is not None else 0,
                "finish_reason": speculative.finish_reason if speculative is not None else "unknown",
                "draft_acceptance_ratio": (
                    speculative.draft_acceptance_ratio if speculative is not None else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_predictions(
    predictions: Iterable[BenchmarkPrediction],
) -> dict[str, dict[str, float | int]]:
    from whispbook_gender_specdec.results import summarize_results_dataframe

    summary_rows = summarize_results_dataframe(predictions_to_dataframe(predictions))
    summary: dict[str, dict[str, float | int]] = {}
    for row in summary_rows.to_dict(orient="records"):
        mode = str(row.pop("mode"))
        summary[mode] = {
            "total": int(row["total"]),
            "correct": int(row["correct"]),
            "accuracy": float(row["accuracy"]),
            "parse_ok_rate": float(row["parse_ok_rate"]),
            "avg_wall_ms": float(row["avg_wall_ms"]),
            "avg_draft_acceptance_ratio": float(row["avg_draft_acceptance_ratio"]),
        }
    return summary


def _avg_draft_acceptance_ratio(rows: list[BenchmarkPrediction]) -> float:
    if not rows:
        return 0.0
    ratios = [
        row.result.speculative.draft_acceptance_ratio
        for row in rows
        if row.result.speculative is not None
    ]
    return sum(ratios) / len(ratios) if ratios else 0.0


def _benchmark_result_row(
    *,
    mode: ModeConfig,
    sample: pd.Series,
    result: GenderInferenceResult,
    wall_ms: float,
    parameters: Mapping[str, object],
) -> dict[str, object]:
    speculative = result.speculative
    row: dict[str, object] = {
        "mode": mode.name,
        "sample_id": _string_cell(sample["sample_id"]),
        "speaker": _string_cell(sample["speaker"]),
        "expected_gender": _string_cell(sample["expected_gender"]),
        "text": _string_cell(sample["text"]),
        "predicted_gender": result.gender,
        "confidence": result.confidence,
        "evidence": result.evidence,
        "raw_response": result.raw_response,
        "parse_ok": result.raw_response.strip().startswith("{"),
        "correct": result.gender == _string_cell(sample["expected_gender"]),
        "wall_ms": wall_ms,
        "target_family": mode.target.family,
        "target_label": mode.target.label,
        "target_model_ref": mode.target.model_ref or "",
        "draft_family": mode.draft.family if mode.draft is not None else "",
        "draft_label": mode.draft.label if mode.draft is not None else "",
        "draft_model_ref": mode.draft.model_ref if mode.draft is not None else "",
        "draft_mode": speculative.draft_mode if speculative is not None else "none",
        "total_tokens": speculative.total_tokens if speculative is not None else 0,
        "draft_tokens": speculative.draft_tokens if speculative is not None else 0,
        "finish_reason": speculative.finish_reason if speculative is not None else "unknown",
        "draft_acceptance_ratio": speculative.draft_acceptance_ratio if speculative is not None else 0.0,
    }
    for column, value in sample.items():
        if column not in row:
            row[column] = _csv_cell(value)
    for key, value in parameters.items():
        column = str(key)
        if column in row:
            column = f"param_{column}"
        row[column] = _csv_cell(value)
    return row


def _attach_metadata_columns(samples: pd.DataFrame, metadata: Mapping[str, object]) -> None:
    title = metadata.get("title")
    if title is not None and "story_title" not in samples.columns:
        samples["story_title"] = _string_cell(title)
    for key in ("target_word_count", "actual_word_count", "sample_count"):
        if key in metadata and key not in samples.columns:
            samples[key] = metadata[key]


def _first_present(sample: pd.Series, *columns: str) -> str:
    for column in columns:
        if column not in sample:
            continue
        value = _string_cell(sample.get(column, ""))
        if value:
            return value
    return ""


def _string_cell(value: object) -> str:
    if _is_missing(value):
        return ""
    return str(value)


def _csv_cell(value: object) -> object:
    if _is_missing(value):
        return ""
    if isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, tuple | list | dict):
        return json.dumps(value, ensure_ascii=True)
    return str(value)


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, bool):
        return missing
    try:
        return bool(missing)
    except (TypeError, ValueError):
        return False
