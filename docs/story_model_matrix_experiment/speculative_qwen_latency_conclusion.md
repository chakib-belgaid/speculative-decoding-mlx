# Speculative Qwen Latency Conclusion

## Conclusion

`Qwen3 4B + Qwen3 0.6B` speculative decoding was slower than `Qwen3 4B` large-only in this experiment. The benchmark measured full per-sample inference cost, while the task produced only a very short JSON classification; these are important confounders rather than separately measured causes.

Speculative decoding matched the target predictions on the tested samples:

| Mode | Accuracy | Avg wall ms | Median wall ms | Avg output tokens | Draft acceptance |
| --- | ---: | ---: | ---: | ---: | ---: |
| `large_only` Qwen3 4B | 97.5% | 1,247 | 1,106 | 29.9 | 0.0% |
| `speculative` Qwen3 4B + Qwen3 0.6B | 97.5% | 1,599 | 1,561 | 29.9 | 75.6% |

The speculative run accepted many draft tokens, but it was still 28.3% slower than the matching large-only baseline.

## Why This Happened

The benchmark timer starts before the inference function call. Inside that call, the current runner loads the target model every time. In speculative mode, it also loads the draft model every time.

So the speculative path pays for:

- target model setup,
- draft model setup,
- draft token generation,
- target verification,
- the same short JSON response generation and parsing.

For this task, both large-only and speculative produced about 29.9 output tokens per sample. The measurements do not isolate setup and decoding costs, so they cannot determine the break-even output length.

## What This Result Means

This result does not mean speculative decoding is useless. It means this benchmark shape is not a valid speedup benchmark yet.

The current result is still useful because it shows:

- Qwen speculative decoding is tokenizer-compatible and executes successfully.
- Speculative predictions match the Qwen large-only predictions.
- Draft acceptance is high enough to justify a better latency experiment.
- The short JSON classifier workload is better suited as a correctness and wiring smoke test.

## What The Next Experiment Must Change

To measure actual speculative speedup, the next experiment should:

1. Load target and draft models once, outside the timed sample loop.
2. Add warmup calls before measurement.
3. Separate load/setup time from decode-only time.
4. Use longer outputs where decode work dominates setup overhead.
5. Run repeated trials and report median, p95, and variance.
6. Track accepted and rejected draft tokens directly.
7. Compare short-output classifier latency separately from long-output generation latency.

The next claim should be precise:

- "Speculative matched target predictions on the tested samples" is supported for the Qwen pair in this run; this does not prove general distributional equivalence.
- "Speculative is faster" is not supported yet and needs a warm, decode-focused benchmark.
