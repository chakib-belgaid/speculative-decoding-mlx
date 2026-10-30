from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from whispbook_gender_specdec.benchmark import (
    load_builtin_story_samples_dataframe,
    load_story_benchmark_dataframe,
    run_model_benchmark_dataframe,
    write_benchmark_csv,
)
from whispbook_gender_specdec.checkpoints import TARGET_FAMILIES, discover_model_specs
from whispbook_gender_specdec.mlx_runner import infer_with_mlx
from whispbook_gender_specdec.results import summarize_results_dataframe
from whispbook_gender_specdec.speculative import choose_draft_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Speculative gender inference and benchmark runner.")
    parser.add_argument("--target", choices=TARGET_FAMILIES, default="qwen_large")
    parser.add_argument("--text", required=False)
    parser.add_argument("--source-hint", default=None)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--num-draft-tokens", type=int, default=3)
    parser.add_argument("--list-checkpoints", action="store_true")
    parser.add_argument("--local-only", action="store_true", help="Only use already cached local checkpoints.")
    parser.add_argument("--benchmark", action="store_true", help="Run the built-in Project Gutenberg story benchmark.")
    parser.add_argument("--benchmark-limit", type=int, default=None)
    parser.add_argument("--ground-truth", type=Path, default=None, help="JSONL benchmark rows with sample_id, speaker, expected_gender, and text.")
    parser.add_argument("--story-text", type=Path, default=None, help="Optional full story text file for custom benchmark context.")
    parser.add_argument("--metadata", type=Path, default=None, help="Optional benchmark metadata JSON file.")
    parser.add_argument("--output-csv", type=Path, default=None, help="Write the full benchmark result dataframe to CSV.")
    parser.add_argument("--small-family", choices=TARGET_FAMILIES, default="qwen_small")
    parser.add_argument("--large-family", choices=TARGET_FAMILIES, default="qwen_large")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    manifest = discover_model_specs(local_only=args.local_only)
    if args.list_checkpoints:
        print(json.dumps({key: asdict(value) for key, value in manifest.items()}, indent=2))
        return 0

    if args.benchmark:
        if (args.story_text is not None or args.metadata is not None) and args.ground_truth is None:
            parser.error("--story-text and --metadata require --ground-truth")
        if args.ground_truth is not None:
            samples = load_story_benchmark_dataframe(
                ground_truth_path=args.ground_truth,
                story_text_path=args.story_text,
                metadata_path=args.metadata,
            )
        else:
            samples = load_builtin_story_samples_dataframe()
        if args.benchmark_limit is not None:
            samples = samples.head(args.benchmark_limit)

        results = run_model_benchmark_dataframe(
            samples=samples,
            manifest=manifest,
            infer_model=infer_with_mlx,
            small_model_name=args.small_family,
            large_model_name=args.large_family,
            max_tokens=args.max_tokens,
            num_draft_tokens=args.num_draft_tokens,
        )
        csv_path = write_benchmark_csv(results, args.output_csv) if args.output_csv is not None else None
        summary = summarize_results_dataframe(results)
        print(
            json.dumps(
                {
                    "summary": _summary_dict(summary),
                    "predictions": _dataframe_records(results),
                    **({"csv_path": str(csv_path)} if csv_path is not None else {}),
                },
                indent=2,
            )
        )
        return 0

    if not args.text:
        parser.error("--text is required unless --list-checkpoints is used")

    target = manifest[args.target]
    draft = choose_draft_checkpoint(target, manifest.values())
    result = infer_with_mlx(
        target=target,
        draft=draft.spec if draft.mode != "unavailable" else None,
        text=args.text,
        source_hint=args.source_hint,
        max_tokens=args.max_tokens,
        num_draft_tokens=args.num_draft_tokens,
    )
    print(json.dumps(asdict(result), indent=2))
    return 0


def _dataframe_records(frame):
    return json.loads(frame.to_json(orient="records"))


def _summary_dict(frame):
    records = _dataframe_records(frame)
    return {str(record.pop("mode")): record for record in records}


if __name__ == "__main__":
    raise SystemExit(main())
