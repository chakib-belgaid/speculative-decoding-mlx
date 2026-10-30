# Story Model Matrix Experiment Summary

Generated: 2026-06-02

This folder packages the comparative analysis for the current Whispbook gender speculative-decoding benchmark. It is intended as the handoff for the next experiment, so it captures what was measured, what was observed, what did not work yet, and what should change before the next run.

## Contents

- [comparative_report.md](comparative_report.md): full comparative report generated from the row-level CSV.
- [speculative_qwen_latency_conclusion.md](speculative_qwen_latency_conclusion.md): focused conclusion explaining why Qwen speculative decoding was slower than Qwen large-only in this run.
- [tables/](tables/): copied CSV summaries used by the report.
- [assets/](assets/): copied PNG charts for quick visual inspection.
- [scripts/generate_story_model_matrix_report.py](scripts/generate_story_model_matrix_report.py): local copy of the report generator used to produce the comparative report.

Key charts:

- [Accuracy by story](assets/story_model_accuracy_heatmap.png)
- [Accuracy versus latency](assets/accuracy_latency_scatter.png)
- [Parse success by configuration](assets/parse_ok_by_config.png)
- [Class-wise accuracy](assets/class_accuracy_heatmap.png)

## Source Data

Primary input:

- package-local copy: [tables/story_model_matrix_predictions.csv](tables/story_model_matrix_predictions.csv)

Copied inputs and derived tables:

- [tables/story_model_matrix_predictions.csv](tables/story_model_matrix_predictions.csv)
- [tables/story_model_matrix_summary.csv](tables/story_model_matrix_summary.csv)
- [tables/overall_config_summary.csv](tables/overall_config_summary.csv)
- [tables/per_story_metrics.csv](tables/per_story_metrics.csv)
- [tables/class_accuracy.csv](tables/class_accuracy.csv)
- [tables/speculative_vs_target.csv](tables/speculative_vs_target.csv)

The matrix contains 680 prediction rows over 5 story fixtures and 40 unique labeled samples. Each story contributes 8 labeled samples. Standalone small and large model rows are repeated across pair labels in the matrix, so the report keeps both raw row counts and unique sample counts visible.

## What We Measured

The benchmark compared:

- `small_only`: small model used directly as the classifier.
- `large_only`: large target model used directly as the classifier.
- `speculative`: large target model verifies draft tokens proposed by a smaller compatible draft model.

The important metrics were:

- Accuracy against labeled speaker-gender ground truth.
- JSON parse success.
- Wall-clock time per sample.
- Draft acceptance ratio for speculative runs.
- Story-level and class-level behavior.

## Main Observations

Qwen3 4B is the best accuracy baseline. It reached 97.5% overall accuracy with 100.0% parse success. It was perfect on stories 2-5 and missed one unknown narrator sample in story 1.

Qwen speculative decoding preserved target behavior but did not improve latency. `Qwen3 4B + Qwen3 0.6B` matched Qwen3 4B predictions and accuracy, accepted about 75.6% of draft tokens, but was 28.3% slower than the matching large-only baseline.

Llama speculative decoding behaved the same way at a lower quality level. `Llama 3.2 3B + SmolLM3 3B` matched Llama large-only behavior exactly, but was 92.4% slower.

SmolLM3 3B is the strongest standalone small model in this run. It reached 80.0% overall accuracy with 100.0% parse success, but it failed badly on story 4 at 25.0% accuracy.

Qwen3 0.6B is the fastest usable model, averaging about 459 ms per sample, but its overall accuracy was only 52.5%. It particularly failed male speaker attribution, with 0.0% class-wise male accuracy in this matrix.

FunctionGemma 270M is not currently viable for this JSON classifier path. It reached only 5.0% parse success and 35.0% overall accuracy, mostly by returning or falling back to `unknown`.

Mistral 7B appears only as a pair label for `small_only` rows in this CSV. There are no Mistral `large_only` or Mistral speculative target rows to evaluate here.

## Why Speculative Was Slower Here

This run timed the whole per-sample inference call, not only steady-state decode. The current runner loads the target model inside each inference call. Speculative mode also loads the draft model inside each inference call. The timing does not separate setup, draft generation, and verification, so their individual contributions cannot be established from this run.

The Qwen pair generated about the same number of output tokens in both modes: around 29.9 tokens per sample. Speculative accepted draft tokens, but this did not translate into a wall-clock improvement in the measured protocol.

See [speculative_qwen_latency_conclusion.md](speculative_qwen_latency_conclusion.md) for the focused conclusion.

## Implications

Speculative decoding matched the target predictions for the tested pairs and samples. This is an integration result on a small dataset, not proof of general distributional equivalence.

This benchmark should not be used to claim speculative decoding speedup. The timing design and the task shape both bias against speculative latency gains:

- Models are loaded inside the measured call.
- Outputs are short JSON objects.
- Each sample is run individually.
- There is no warm steady-state decode measurement.
- There are no repeated trials or confidence intervals.

## Next Experiment Plan

The next experiment should separate model quality from runtime mechanics.

Recommended benchmark changes:

1. Cache loaded models and tokenizers outside the per-sample loop.
2. Split timing into load time, prompt construction, tokenization/prefill, decode, parsing, and total wall time.
3. Add warmup calls before measuring.
4. Run repeated trials per sample and report mean, median, p95, and variance.
5. Use longer generations where speculative decoding has enough decode work to amortize draft overhead.
6. Track generated target tokens, accepted draft tokens, rejected draft tokens, and tokens per second.
7. Keep the current short JSON classifier benchmark as a correctness and wiring test, not as the main speed benchmark.
8. Increase the number of labeled samples per story before treating story-level differences as stable.

Recommended model set:

- Accuracy baseline: `qwen_large`.
- Speculative pair: `qwen_large + qwen_small`.
- Fast small baseline: `qwen_small`.
- Strong standalone small baseline: `smollm`.
- Keep `llama_3_2_3b + smollm` as a secondary compatibility check.
- Exclude FunctionGemma from speed comparisons until JSON parsing is fixed.

Recommended success criteria:

- Speculative output matches target output or target accuracy stays within an agreed tolerance.
- Speculative decode-only latency beats large-only decode-only latency after warmup.
- End-to-end latency is reported separately from decode-only latency.
- Short-output and long-output workloads are reported separately.

## Reproducibility

The report was generated with:

```bash
uv run --frozen python scripts/generate_story_model_matrix_report.py \
  --csv docs/story_model_matrix_experiment/tables/story_model_matrix_predictions.csv \
  --report results/comparative_report.md \
  --assets-dir results/assets
```

The generator script is:

- [scripts/generate_story_model_matrix_report.py](scripts/generate_story_model_matrix_report.py)
