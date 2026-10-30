from whispbook_gender_specdec.benchmark import (
    BenchmarkPrediction,
    default_mode_configs,
    load_ground_truth_dataframe,
    load_builtin_story_samples,
    run_model_benchmark_dataframe,
    run_benchmark,
    run_benchmark_dataframe,
    summarize_predictions,
    write_benchmark_csv,
)
from whispbook_gender_specdec.checkpoints import discover_model_specs
from whispbook_gender_specdec.gender import GenderInferenceResult
from whispbook_gender_specdec.results import summarize_results_dataframe
from whispbook_gender_specdec.results import summarize_story_model_matrix_dataframe
from whispbook_gender_specdec.speculative import SpeculativeStats


def test_builtin_story_samples_cover_multiple_speakers_and_genders() -> None:
    samples = load_builtin_story_samples()

    assert len(samples) >= 8
    assert {sample.source.project_gutenberg_id for sample in samples} == {2542}
    assert {sample.expected_gender for sample in samples} >= {"male", "female"}
    assert len({sample.speaker for sample in samples}) >= 4


def test_load_ground_truth_dataframe_reads_jsonl_samples(tmp_path) -> None:
    ground_truth = tmp_path / "ground_truth.jsonl"
    ground_truth.write_text(
        "\n".join(
            [
                '{"sample_id":"s1","speaker":"Mara","expected_gender":"female","text":"Mara said hello."}',
                '{"sample_id":"s2","speaker":"narrator","expected_gender":"unknown","text":"Rain fell."}',
            ]
        ),
        encoding="utf-8",
    )

    samples = load_ground_truth_dataframe(ground_truth)

    assert list(samples.columns)[:4] == ["sample_id", "speaker", "expected_gender", "text"]
    assert samples["sample_id"].tolist() == ["s1", "s2"]
    assert samples["expected_gender"].tolist() == ["female", "unknown"]


def test_default_mode_configs_compare_small_large_and_speculative() -> None:
    manifest = discover_model_specs(local_only=False)

    modes = default_mode_configs(manifest)

    assert [mode.name for mode in modes] == ["small_only", "large_only", "speculative"]
    assert modes[0].target.family == "qwen_small"
    assert modes[0].draft is None
    assert modes[1].target.family == "qwen_large"
    assert modes[1].draft is None
    assert modes[2].target.family == "qwen_large"
    assert modes[2].draft is not None
    assert modes[2].draft.family == "qwen_small"


def test_summarize_predictions_reports_accuracy_and_latency_by_mode() -> None:
    predictions = [
        _prediction("small_only", "female", "female", 100.0),
        _prediction("small_only", "female", "male", 140.0),
        _prediction("large_only", "male", "male", 220.0),
        _prediction("speculative", "female", "female", 160.0, draft_tokens=8),
    ]

    summary = summarize_predictions(predictions)

    assert summary["small_only"]["total"] == 2
    assert summary["small_only"]["accuracy"] == 0.5
    assert summary["small_only"]["avg_wall_ms"] == 120.0
    assert summary["large_only"]["accuracy"] == 1.0
    assert summary["speculative"]["avg_draft_acceptance_ratio"] == 0.8


def test_summarize_results_dataframe_reports_accuracy_and_latency_by_mode() -> None:
    results = run_benchmark_dataframe(
        samples=_sample_dataframe(),
        modes=_fake_modes()[:2],
        infer=lambda mode, row: _result(row["expected_gender"] if mode.name == "small_only" else "unknown"),
        parameters={"max_tokens": 64, "num_draft_tokens": 3},
    )

    summary = summarize_results_dataframe(results)

    assert summary["mode"].tolist() == ["large_only", "small_only"]
    large = summary.set_index("mode").loc["large_only"]
    small = summary.set_index("mode").loc["small_only"]
    assert large["accuracy"] == 0.5
    assert small["accuracy"] == 1.0
    assert small["total"] == 2


def test_summarize_results_dataframe_handles_csv_loaded_boolean_strings() -> None:
    import pandas as pd

    results = pd.DataFrame(
        [
            {
                "mode": "small_only",
                "sample_id": "s1",
                "correct": "True",
                "parse_ok": "true",
                "wall_ms": 10.0,
                "draft_acceptance_ratio": 0.0,
            },
            {
                "mode": "small_only",
                "sample_id": "s2",
                "correct": "False",
                "parse_ok": "false",
                "wall_ms": 30.0,
                "draft_acceptance_ratio": 1.0,
            },
        ]
    )

    summary = summarize_results_dataframe(results)

    row = summary.set_index("mode").loc["small_only"]
    assert row["accuracy"] == 0.5
    assert row["parse_ok_rate"] == 0.5


def test_summarize_story_model_matrix_dataframe_groups_by_story_and_model_pair() -> None:
    import pandas as pd

    results = pd.DataFrame(
        [
            {
                "story_key": "story_1",
                "story_title": "Glass Archive",
                "small_model_name": "qwen_small",
                "large_model_name": "qwen_large",
                "mode": "speculative",
                "correct": True,
                "parse_ok": True,
                "wall_ms": 120.0,
                "draft_acceptance_ratio": 0.5,
            },
            {
                "story_key": "story_1",
                "story_title": "Glass Archive",
                "small_model_name": "qwen_small",
                "large_model_name": "qwen_large",
                "mode": "speculative",
                "correct": False,
                "parse_ok": True,
                "wall_ms": 180.0,
                "draft_acceptance_ratio": 0.7,
            },
            {
                "story_key": "story_2",
                "story_title": "Ciphered Tide",
                "small_model_name": "gemma",
                "large_model_name": "mistral_7b",
                "mode": "large_only",
                "correct": "true",
                "parse_ok": "true",
                "wall_ms": 300.0,
                "draft_acceptance_ratio": 0.0,
            },
        ]
    )

    summary = summarize_story_model_matrix_dataframe(results)

    assert list(summary.columns) == [
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
    ]
    spec_row = summary.set_index(["story_key", "small_model_name", "large_model_name", "mode"]).loc[
        ("story_1", "qwen_small", "qwen_large", "speculative")
    ]
    assert spec_row["total"] == 2
    assert spec_row["correct"] == 1
    assert spec_row["accuracy"] == 0.5
    assert spec_row["parse_ok_rate"] == 1.0
    assert spec_row["avg_wall_ms"] == 150.0
    assert spec_row["avg_draft_acceptance_ratio"] == 0.6


def test_run_benchmark_calls_each_mode_for_each_sample() -> None:
    manifest = discover_model_specs(local_only=False)
    modes = default_mode_configs(manifest)[:2]
    samples = load_builtin_story_samples()[:2]
    calls: list[tuple[str, str]] = []

    def fake_infer(mode, sample):
        calls.append((mode.name, sample.sample_id))
        return _prediction(mode.name, sample.expected_gender, sample.expected_gender, 0).result

    predictions = run_benchmark(samples, modes, fake_infer)

    assert len(predictions) == 4
    assert calls == [
        ("small_only", samples[0].sample_id),
        ("small_only", samples[1].sample_id),
        ("large_only", samples[0].sample_id),
        ("large_only", samples[1].sample_id),
    ]


def test_run_benchmark_dataframe_returns_full_result_rows() -> None:
    calls: list[tuple[str, str]] = []

    def fake_infer(mode, row):
        calls.append((mode.name, row["sample_id"]))
        return _result(row["expected_gender"])

    results = run_benchmark_dataframe(
        samples=_sample_dataframe(),
        modes=_fake_modes(),
        infer=fake_infer,
        parameters={"max_tokens": 64, "temperature": 0.0},
    )

    assert calls == [
        ("small_only", "sample_1"),
        ("small_only", "sample_2"),
        ("large_only", "sample_1"),
        ("large_only", "sample_2"),
        ("speculative", "sample_1"),
        ("speculative", "sample_2"),
    ]
    assert len(results) == 6
    assert set(
        [
            "mode",
            "sample_id",
            "speaker",
            "expected_gender",
            "text",
            "predicted_gender",
            "correct",
            "wall_ms",
            "target_family",
            "draft_family",
            "draft_acceptance_ratio",
            "max_tokens",
            "temperature",
        ]
    ).issubset(results.columns)
    assert results["correct"].tolist() == [True, True, True, True, True, True]


def test_run_model_benchmark_dataframe_passes_model_names_and_extra_parameters() -> None:
    manifest = discover_model_specs(local_only=False)
    calls: list[dict[str, object]] = []

    def fake_model_runner(**kwargs):
        calls.append(kwargs)
        return _result("female")

    results = run_model_benchmark_dataframe(
        samples=_sample_dataframe().head(1),
        manifest=manifest,
        infer_model=fake_model_runner,
        small_model_name="qwen_small",
        large_model_name="qwen_large",
        max_tokens=32,
        num_draft_tokens=5,
        extra_parameters={"temperature": 0.0},
    )

    assert results["mode"].tolist() == ["small_only", "large_only", "speculative"]
    assert results["small_model_name"].tolist() == ["qwen_small", "qwen_small", "qwen_small"]
    assert results["large_model_name"].tolist() == ["qwen_large", "qwen_large", "qwen_large"]
    assert all(call["max_tokens"] == 32 for call in calls)
    assert all(call["num_draft_tokens"] == 5 for call in calls)
    assert all(call["temperature"] == 0.0 for call in calls)
    assert calls[0]["draft"] is None
    assert calls[2]["draft"].family == "qwen_small"


def test_write_benchmark_csv_round_trips_dataframe(tmp_path) -> None:
    results = run_benchmark_dataframe(
        samples=_sample_dataframe().head(1),
        modes=_fake_modes()[:1],
        infer=lambda mode, row: _result(row["expected_gender"]),
    )
    csv_path = tmp_path / "results.csv"

    written = write_benchmark_csv(results, csv_path)

    assert written == csv_path
    assert "sample_1" in csv_path.read_text(encoding="utf-8")


def _prediction(
    mode: str,
    expected: str,
    predicted: str,
    wall_ms: float,
    draft_tokens: int = 0,
) -> BenchmarkPrediction:
    return BenchmarkPrediction(
        mode=mode,
        sample_id=f"{mode}_{expected}_{predicted}",
        speaker="NORA",
        expected_gender=expected,
        result=GenderInferenceResult(
            gender=predicted,
            confidence=0.7,
            evidence="fixture",
            raw_response="{}",
            speculative=SpeculativeStats(
                target_family="target",
                draft_family="draft" if draft_tokens else "none",
                draft_mode="external_draft" if draft_tokens else "none",
                total_tokens=10,
                draft_tokens=draft_tokens,
                finish_reason="stop",
            ),
        ),
        wall_ms=wall_ms,
    )


def _sample_dataframe():
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "sample_id": "sample_1",
                "speaker": "Mara",
                "expected_gender": "female",
                "text": "Mara said hello.",
                "story_title": "Fixture Story",
            },
            {
                "sample_id": "sample_2",
                "speaker": "narrator",
                "expected_gender": "unknown",
                "text": "Rain fell.",
                "story_title": "Fixture Story",
            },
        ]
    )


def _fake_modes():
    manifest = discover_model_specs(local_only=False)
    return default_mode_configs(manifest)


def _result(predicted: str) -> GenderInferenceResult:
    return GenderInferenceResult(
        gender=predicted,
        confidence=0.9,
        evidence="fixture",
        raw_response='{"gender":"%s"}' % predicted,
        speculative=SpeculativeStats(
            target_family="target",
            draft_family="draft",
            draft_mode="external_draft",
            total_tokens=10,
            draft_tokens=5,
            finish_reason="stop",
        ),
    )
