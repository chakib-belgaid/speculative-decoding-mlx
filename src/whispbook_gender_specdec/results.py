from __future__ import annotations

import pandas as pd


SUMMARY_COLUMNS = (
    "mode",
    "total",
    "correct",
    "accuracy",
    "parse_ok_rate",
    "avg_wall_ms",
    "avg_draft_acceptance_ratio",
)
STORY_MODEL_MATRIX_COLUMNS = (
    "story_key",
    "story_title",
    "small_model_name",
    "large_model_name",
    "mode",
    "total",
    "correct",
    "accuracy",
    "parse_ok_rate",
    "avg_wall_ms",
    "avg_draft_acceptance_ratio",
)


def summarize_results_dataframe(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    _require_columns(results, ["mode", "correct", "wall_ms"])
    frame = results.copy()
    frame["correct"] = _bool_series(frame["correct"])
    frame["_parse_ok"] = _parse_ok_series(frame)
    if "draft_acceptance_ratio" not in frame.columns:
        frame["draft_acceptance_ratio"] = 0.0

    grouped = frame.groupby("mode", sort=True)
    summary = grouped.agg(
        total=("mode", "size"),
        correct=("correct", "sum"),
        parse_ok_rate=("_parse_ok", "mean"),
        avg_wall_ms=("wall_ms", "mean"),
        avg_draft_acceptance_ratio=("draft_acceptance_ratio", "mean"),
    ).reset_index()
    summary["accuracy"] = summary["correct"] / summary["total"]
    return summary.loc[:, SUMMARY_COLUMNS]


def summarize_results_by_speaker_dataframe(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(
            columns=[
                "mode",
                "speaker",
                "expected_gender",
                "total",
                "correct",
                "accuracy",
                "avg_wall_ms",
            ]
        )

    _require_columns(results, ["mode", "speaker", "expected_gender", "correct", "wall_ms"])
    frame = results.copy()
    frame["correct"] = _bool_series(frame["correct"])
    summary = (
        frame.groupby(["mode", "speaker", "expected_gender"], sort=True)
        .agg(
            total=("mode", "size"),
            correct=("correct", "sum"),
            avg_wall_ms=("wall_ms", "mean"),
        )
        .reset_index()
    )
    summary["accuracy"] = summary["correct"] / summary["total"]
    return summary.loc[
        :,
        ["mode", "speaker", "expected_gender", "total", "correct", "accuracy", "avg_wall_ms"],
    ]


def summarize_story_model_matrix_dataframe(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(columns=STORY_MODEL_MATRIX_COLUMNS)

    group_columns = ["story_key", "story_title", "small_model_name", "large_model_name", "mode"]
    _require_columns(results, [*group_columns, "correct", "wall_ms"])
    frame = results.copy()
    for column in group_columns:
        frame[column] = frame[column].fillna("").astype(str)
    frame["correct"] = _bool_series(frame["correct"])
    frame["_parse_ok"] = _parse_ok_series(frame)
    if "draft_acceptance_ratio" not in frame.columns:
        frame["draft_acceptance_ratio"] = 0.0

    summary = (
        frame.groupby(group_columns, sort=True, dropna=False)
        .agg(
            total=("mode", "size"),
            correct=("correct", "sum"),
            parse_ok_rate=("_parse_ok", "mean"),
            avg_wall_ms=("wall_ms", "mean"),
            avg_draft_acceptance_ratio=("draft_acceptance_ratio", "mean"),
        )
        .reset_index()
    )
    summary["accuracy"] = summary["correct"] / summary["total"]
    return summary.loc[:, STORY_MODEL_MATRIX_COLUMNS]


def _parse_ok_series(results: pd.DataFrame) -> pd.Series:
    if "parse_ok" in results.columns:
        return _bool_series(results["parse_ok"])
    if "raw_response" not in results.columns:
        return pd.Series([False] * len(results), index=results.index)
    return results["raw_response"].map(lambda value: str(value).strip().startswith("{"))


def _bool_series(series: pd.Series) -> pd.Series:
    if str(series.dtype) == "bool":
        return series.fillna(False)
    return series.map(_bool_cell)


def _bool_cell(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return bool(value)


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"results dataframe is missing required columns: {', '.join(missing)}")
