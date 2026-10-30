import json
from pathlib import Path

from whispbook_gender_specdec.checkpoints import TARGET_FAMILIES, discover_local_checkpoints
from whispbook_gender_specdec.cli import build_parser


def test_declares_requested_checkpoint_families() -> None:
    assert set(TARGET_FAMILIES) == {
        "smollm",
        "gemma",
        "qwen",
        "qwen_small",
        "qwen_large",
        "llama_3_2_3b",
        "mistral_7b",
    }


def test_smollm_discovery_checks_smollm2_and_smollm3_candidates(tmp_path) -> None:
    manifest = discover_local_checkpoints(hf_cache=tmp_path / "empty-hf-cache")

    notes = "\n".join(manifest["smollm"].notes)
    assert "SmolLM2" in notes
    assert "SmolLM3" in notes


def test_discovers_local_gemma_and_qwen_without_remote_paths(tmp_path) -> None:
    for repo_id, vocab_size in (
        ("mlx-community/functiongemma-270m-it-4bit", 262144),
        ("mlx-community/Qwen3-4B-Instruct-2507-4bit", 151936),
    ):
        snapshot = tmp_path / f"models--{repo_id.replace('/', '--')}" / "snapshots" / "test-revision"
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text(json.dumps({"vocab_size": vocab_size}))
        (snapshot / "tokenizer.json").write_text("{}")
        (snapshot / "model.safetensors").touch()

    manifest = discover_local_checkpoints(hf_cache=tmp_path)

    assert set(manifest) == set(TARGET_FAMILIES)
    assert manifest["gemma"].available, manifest["gemma"].notes
    assert manifest["qwen"].available, manifest["qwen"].notes
    assert manifest["gemma"].vocab_size == 262144
    assert manifest["qwen"].vocab_size == 151936
    assert not manifest["smollm"].available
    assert manifest["smollm"].source != "remote"

    for spec in manifest.values():
        if spec.path is None:
            continue
        assert Path(spec.path).exists()
        assert "://" not in spec.path


def test_incomplete_cached_checkpoint_is_not_available(tmp_path) -> None:
    snapshot = tmp_path / "models--mlx-community--Qwen3-4B-Instruct-2507-4bit" / "snapshots" / "incomplete"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text('{"vocab_size": 151936}')
    (snapshot / "tokenizer.json").write_text("{}")

    manifest = discover_local_checkpoints(hf_cache=tmp_path)

    assert not manifest["qwen"].available
    assert manifest["qwen"].path is None
    assert any("not a standalone local checkpoint" in note for note in manifest["qwen"].notes)


def test_cli_accepts_all_requested_checkpoint_families() -> None:
    parser = build_parser()

    for target in TARGET_FAMILIES:
        args = parser.parse_args(["--target", target, "--text", "hello"])
        assert args.target == target


def test_cli_exposes_story_benchmark_flags() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "--benchmark",
            "--benchmark-limit",
            "2",
            "--ground-truth",
            "fixtures.jsonl",
            "--story-text",
            "story.txt",
            "--metadata",
            "metadata.json",
            "--output-csv",
            "results.csv",
        ]
    )

    assert args.benchmark is True
    assert args.benchmark_limit == 2
    assert str(args.ground_truth) == "fixtures.jsonl"
    assert str(args.story_text) == "story.txt"
    assert str(args.metadata) == "metadata.json"
    assert str(args.output_csv) == "results.csv"
