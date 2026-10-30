from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

VALID_GENDERS = {"female", "male", "unknown"}
REQUIRED_ROW_KEYS = {"sample_id", "speaker", "expected_gender", "text"}


class FixtureError(ValueError):
    pass


def validate_story_dir(story_dir: Path) -> None:
    if not story_dir.is_dir():
        raise FixtureError(f"{story_dir} is not a directory")

    story_file = _one_file(story_dir, "*_20000_word_story.txt")
    jsonl_file = _one_file(story_dir, "*_speaker_attribution_ground_truth.jsonl")
    metadata_file = _one_file(story_dir, "*_benchmark_metadata.json")
    slug = story_file.name.removesuffix("_20000_word_story.txt")

    if jsonl_file.name != f"{slug}_speaker_attribution_ground_truth.jsonl":
        raise FixtureError(f"{story_dir}: JSONL filename does not match slug {slug}")
    if metadata_file.name != f"{slug}_benchmark_metadata.json":
        raise FixtureError(f"{story_dir}: metadata filename does not match slug {slug}")

    rows = _load_jsonl(jsonl_file)
    metadata = _load_metadata(metadata_file)
    story_text = story_file.read_text(encoding="utf-8").strip()

    _validate_rows(story_dir, slug, rows)
    _validate_metadata(story_dir, metadata, rows, story_text)


def _one_file(story_dir: Path, pattern: str) -> Path:
    matches = sorted(story_dir.glob(pattern))
    if len(matches) != 1:
        raise FixtureError(f"{story_dir}: expected exactly one {pattern}, found {len(matches)}")
    return matches[0]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise FixtureError(f"{path}:{line_number}: blank JSONL lines are not allowed")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FixtureError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise FixtureError(f"{path}:{line_number}: row must be an object")
        rows.append(row)
    if not rows:
        raise FixtureError(f"{path}: no rows found")
    return rows


def _load_metadata(path: Path) -> dict[str, Any]:
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FixtureError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(metadata, dict):
        raise FixtureError(f"{path}: metadata must be an object")
    return metadata


def _validate_rows(story_dir: Path, slug: str, rows: list[dict[str, Any]]) -> None:
    for index, row in enumerate(rows, start=1):
        missing = REQUIRED_ROW_KEYS - row.keys()
        if missing:
            raise FixtureError(f"{story_dir}: row {index} missing keys {sorted(missing)}")
        for key in REQUIRED_ROW_KEYS:
            if not isinstance(row[key], str) or not row[key].strip():
                raise FixtureError(f"{story_dir}: row {index} field {key} must be a non-empty string")

        expected_sample_id = f"{slug}_{index:04d}"
        if row["sample_id"] != expected_sample_id:
            raise FixtureError(
                f"{story_dir}: row {index} sample_id {row['sample_id']!r} should be {expected_sample_id!r}"
            )
        if row["expected_gender"] not in VALID_GENDERS:
            raise FixtureError(f"{story_dir}: row {index} has invalid gender {row['expected_gender']!r}")
        if row["speaker"] == "narrator" and row["expected_gender"] != "unknown":
            raise FixtureError(f"{story_dir}: narrator row {index} must have unknown gender")


def _validate_metadata(
    story_dir: Path,
    metadata: dict[str, Any],
    rows: list[dict[str, Any]],
    story_text: str,
) -> None:
    for key in ("title", "target_word_count", "actual_word_count", "sample_count", "labels", "notes"):
        if key not in metadata:
            raise FixtureError(f"{story_dir}: metadata missing {key}")

    if metadata["sample_count"] != len(rows):
        raise FixtureError(
            f"{story_dir}: metadata sample_count {metadata['sample_count']} != {len(rows)} rows"
        )

    actual_word_count = len(story_text.split())
    if metadata["actual_word_count"] != actual_word_count:
        raise FixtureError(
            f"{story_dir}: metadata actual_word_count {metadata['actual_word_count']} != {actual_word_count}"
        )

    joined_rows = "\n\n".join(row["text"].strip() for row in rows)
    if story_text != joined_rows:
        raise FixtureError(f"{story_dir}: story text does not match ordered JSONL row text")

    labels = metadata["labels"]
    if not isinstance(labels, dict):
        raise FixtureError(f"{story_dir}: metadata labels must be an object")

    speakers = {row["speaker"] for row in rows}
    missing_speakers = speakers - labels.keys()
    if missing_speakers:
        raise FixtureError(f"{story_dir}: metadata labels missing speakers {sorted(missing_speakers)}")

    for speaker in speakers:
        label = labels[speaker]
        if not isinstance(label, dict):
            raise FixtureError(f"{story_dir}: label for {speaker!r} must be an object")
        if label.get("expected_gender") not in VALID_GENDERS:
            raise FixtureError(f"{story_dir}: label for {speaker!r} has invalid expected_gender")
        row_genders = {row["expected_gender"] for row in rows if row["speaker"] == speaker}
        if row_genders != {label["expected_gender"]}:
            raise FixtureError(
                f"{story_dir}: label gender for {speaker!r} does not match row genders {sorted(row_genders)}"
            )


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("usage: validate_story_fixtures.py STORY_DIR [STORY_DIR ...]", file=sys.stderr)
        return 2

    for raw_path in args:
        story_dir = Path(raw_path)
        try:
            validate_story_dir(story_dir)
        except FixtureError as exc:
            print(f"ERROR {exc}", file=sys.stderr)
            return 1
        print(f"OK {story_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
