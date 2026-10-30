# Story Model Matrix Comparative Report

Generated: 2026-06-02 18:32

Source CSV: `notebooks/outputs/story_model_matrix_predictions.csv`

## Scope

This report analyzes 680 prediction rows across 5 story fixtures and 40 unique labeled story samples. Each story contributes 8 labeled samples. Standalone small and large model rows are repeated across model-pair labels in the matrix, so the tables report both raw row counts and unique story samples.

Stories:

- `story_1`: The Glass Archive
- `story_2`: The Ciphered Tide
- `story_3`: The Ashen Lantern
- `story_4`: The Mirrored Depositions
- `story_5`: The Redacted Palace

Unique sample label mix: female=10, male=16, unknown=14.

## Executive Summary

- Best accuracy: Qwen3 4B reached 97.5% overall with 100.0% parse success. It was perfect on stories 2-5 and missed one sample in story 1.
- Speculative decoding preserved target quality but did not speed up this benchmark. `Qwen3 4B + Qwen3 0.6B` matched Qwen3 4B accuracy and predictions, but was +28.3% slower than the matching large-only baseline. `Llama 3.2 3B + SmolLM3 3B` also matched target behavior but was +92.4% slower.
- Best standalone small model: SmolLM3 3B reached 80.0% at 1,130 ms average wall time. It beat Qwen3 0.6B on accuracy but was much slower.
- Fastest usable model: Qwen3 0.6B averaged 459 ms, but its overall accuracy was only 52.5%.
- FunctionGemma is not ready for this JSON classifier path: parse success was only 5.0% and overall accuracy was 35.0%.

## Overall Configuration Ranking

| Configuration | Rows | Unique samples | Accuracy | Parse OK | Avg wall ms | P95 wall ms | Draft acceptance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Large only: Qwen3 4B | 120 | 40 | 97.5% | 100.0% | 1,273 | 1,900 | 0.0% |
| Speculative: Qwen3 4B + Qwen3 0.6B | 40 | 40 | 97.5% | 100.0% | 1,599 | 1,817 | 75.6% |
| Small only: SmolLM3 3B | 120 | 40 | 80.0% | 100.0% | 1,130 | 1,472 | 0.0% |
| Large only: Llama 3.2 3B | 120 | 40 | 62.5% | 100.0% | 1,100 | 1,342 | 0.0% |
| Speculative: Llama 3.2 3B + SmolLM3 3B | 40 | 40 | 62.5% | 100.0% | 2,138 | 2,800 | 77.3% |
| Small only: Qwen3 0.6B | 120 | 40 | 52.5% | 100.0% | 459 | 525 | 0.0% |
| Small only: FunctionGemma 270M | 120 | 40 | 35.0% | 5.0% | 1,264 | 1,360 | 0.0% |

## Accuracy By Story

| Configuration | story_1 | story_2 | story_3 | story_4 | story_5 |
| --- | --- | --- | --- | --- | --- |
| Large only: Qwen3 4B | 87.5% | 100.0% | 100.0% | 100.0% | 100.0% |
| Speculative: Qwen3 4B + Qwen3 0.6B | 87.5% | 100.0% | 100.0% | 100.0% | 100.0% |
| Large only: Llama 3.2 3B | 37.5% | 75.0% | 62.5% | 62.5% | 75.0% |
| Speculative: Llama 3.2 3B + SmolLM3 3B | 37.5% | 75.0% | 62.5% | 62.5% | 75.0% |
| Small only: SmolLM3 3B | 87.5% | 87.5% | 100.0% | 25.0% | 100.0% |
| Small only: Qwen3 0.6B | 75.0% | 37.5% | 50.0% | 37.5% | 62.5% |
| Small only: FunctionGemma 270M | 62.5% | 25.0% | 37.5% | 25.0% | 25.0% |

## Speculative Versus Matching Large-Only Target

Speculative rows are compared to the large-only rows with the same target and draft-family pairing label.

| Pair | Samples | Large acc | Spec acc | Prediction match | Large avg ms | Spec avg ms | Slowdown | Draft acceptance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Llama 3.2 3B + SmolLM3 3B | 40 | 62.5% | 62.5% | 100.0% | 1,112 | 2,138 | +92.4% | 77.3% |
| Qwen3 4B + Qwen3 0.6B | 40 | 97.5% | 97.5% | 100.0% | 1,247 | 1,599 | +28.3% | 75.6% |

## Class-Wise Accuracy

| Configuration | Female | Male | Unknown |
| --- | --- | --- | --- |
| Large only: Qwen3 4B | 100.0% | 100.0% | 92.9% |
| Speculative: Qwen3 4B + Qwen3 0.6B | 100.0% | 100.0% | 92.9% |
| Large only: Llama 3.2 3B | 80.0% | 93.8% | 14.3% |
| Speculative: Llama 3.2 3B + SmolLM3 3B | 80.0% | 93.8% | 14.3% |
| Small only: SmolLM3 3B | 80.0% | 62.5% | 100.0% |
| Small only: Qwen3 0.6B | 70.0% | 0.0% | 100.0% |
| Small only: FunctionGemma 270M | 0.0% | 0.0% | 100.0% |

## Diagnostics

- Repeated standalone row consistency: No prediction, correctness, or parse-status inconsistencies were found across repeated standalone rows.
- Mistral 7B appears only as a `large_model_name` pairing label for `small_only` rows in this CSV. There are no `large_only` or `speculative` Mistral target rows to compare here.
- Draft acceptance is high for both speculative pairs, around 71-80% by story, but the generated outputs are short JSON classifications. The overhead of speculative setup and verification dominates the current wall-clock timings.

Representative failure rows:

| Configuration | Story | Sample | Speaker | Expected | Predicted | Parse OK |
| --- | --- | --- | --- | --- | --- | --- |
| Small only: Qwen3 0.6B | story_1 | glass_archive_0003 | Mara Vale | female | unknown | True |
| Small only: Qwen3 0.6B | story_1 | glass_archive_0004 | Idris Noor | male | female | True |
| Large only: Qwen3 4B | story_1 | glass_archive_0001 | narrator | unknown | female | True |
| Speculative: Qwen3 4B + Qwen3 0.6B | story_1 | glass_archive_0001 | narrator | unknown | female | True |
| Small only: FunctionGemma 270M | story_1 | glass_archive_0003 | Mara Vale | female | unknown | False |
| Small only: FunctionGemma 270M | story_1 | glass_archive_0004 | Idris Noor | male | unknown | False |
| Small only: FunctionGemma 270M | story_1 | glass_archive_0005 | Sera Kade | female | unknown | False |
| Small only: SmolLM3 3B | story_1 | glass_archive_0005 | Sera Kade | female | unknown | True |
| Small only: Qwen3 0.6B | story_2 | ciphered_tide_0003 | Jon Vale | male | unknown | True |
| Small only: Qwen3 0.6B | story_2 | ciphered_tide_0004 | Elias Rook | male | unknown | True |
| Small only: Qwen3 0.6B | story_2 | ciphered_tide_0006 | Oren Pike | male | unknown | True |
| Small only: Qwen3 0.6B | story_2 | ciphered_tide_0007 | Elias Rook | male | unknown | True |

## Visuals

- [Accuracy Heatmap](assets/story_model_accuracy_heatmap.png)
- [Accuracy Latency](assets/accuracy_latency_scatter.png)
- [Parse Ok](assets/parse_ok_by_config.png)
- [Class Accuracy](assets/class_accuracy_heatmap.png)

## Recommendation

Use Qwen3 4B as the accuracy baseline for this task. Keep speculative decoding wired as an integration path, but do not claim a latency win from this run: both speculative pairs are slower than their matching large-only baselines despite high draft acceptance. For a fast fallback, Qwen3 0.6B is the latency winner but needs stronger prompting or a second-stage verifier. SmolLM3 is the strongest standalone small model, though story 4 shows it can fail badly on harder document-frame attribution.
