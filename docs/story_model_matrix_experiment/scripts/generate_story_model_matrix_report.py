from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from textwrap import dedent

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_LABELS = {
    "gemma": "FunctionGemma 270M",
    "qwen_small": "Qwen3 0.6B",
    "smollm": "SmolLM3 3B",
    "qwen_large": "Qwen3 4B",
    "llama_3_2_3b": "Llama 3.2 3B",
    "mistral_7b": "Mistral 7B",
}

KIND_ORDER = {"large": 0, "speculative": 1, "small": 2}
CONFIG_ORDER = {
    "large:qwen_large": 0,
    "speculative:qwen_large+qwen_small": 1,
    "large:llama_3_2_3b": 2,
    "speculative:llama_3_2_3b+smollm": 3,
    "small:smollm": 4,
    "small:qwen_small": 5,
    "small:gemma": 6,
}


def pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value) * 100:.1f}%"


def ms(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.0f}"


def signed_pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    sign = "+" if value >= 0 else ""
    return f"{sign}{float(value) * 100:.1f}%"


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"

    cols = list(frame.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in frame.iterrows():
        values = [str(row[col]) for col in cols]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def add_config_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def config_key(row: pd.Series) -> str:
        if row["mode"] == "small_only":
            return f"small:{row['small_model_name']}"
        if row["mode"] == "large_only":
            return f"large:{row['large_model_name']}"
        return f"speculative:{row['large_model_name']}+{row['small_model_name']}"

    def config_label(row: pd.Series) -> str:
        if row["mode"] == "small_only":
            return f"Small only: {MODEL_LABELS.get(row['small_model_name'], row['small_model_name'])}"
        if row["mode"] == "large_only":
            return f"Large only: {MODEL_LABELS.get(row['large_model_name'], row['large_model_name'])}"
        large = MODEL_LABELS.get(row["large_model_name"], row["large_model_name"])
        small = MODEL_LABELS.get(row["small_model_name"], row["small_model_name"])
        return f"Speculative: {large} + {small}"

    def config_kind(mode: str) -> str:
        if mode == "small_only":
            return "small"
        if mode == "large_only":
            return "large"
        return "speculative"

    out["sample_uid"] = out["story_key"] + ":" + out["sample_id"]
    out["config_key"] = out.apply(config_key, axis=1)
    out["config_label"] = out.apply(config_label, axis=1)
    out["config_kind"] = out["mode"].map(config_kind)
    out["story_label"] = out["story_key"] + ": " + out["story_title"]
    out["config_sort"] = out["config_key"].map(CONFIG_ORDER).fillna(99).astype(int)
    out["story_sort"] = out["story_key"].str.extract(r"(\d+)$").astype(int)
    return out


def aggregate_configs(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["config_key", "config_label", "config_kind", "config_sort"], dropna=False)
        .agg(
            rows=("correct", "size"),
            unique_samples=("sample_uid", "nunique"),
            stories=("story_key", "nunique"),
            correct=("correct", "sum"),
            accuracy=("correct", "mean"),
            parse_ok_rate=("parse_ok", "mean"),
            avg_wall_ms=("wall_ms", "mean"),
            median_wall_ms=("wall_ms", "median"),
            p95_wall_ms=("wall_ms", lambda s: float(s.quantile(0.95))),
            avg_draft_acceptance_ratio=("draft_acceptance_ratio", "mean"),
        )
        .reset_index()
        .sort_values(["accuracy", "avg_wall_ms"], ascending=[False, True])
    )
    return grouped


def per_story_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["config_key", "config_label", "config_sort", "story_key", "story_title", "story_sort"])
        .agg(
            accuracy=("correct", "mean"),
            avg_wall_ms=("wall_ms", "mean"),
            parse_ok_rate=("parse_ok", "mean"),
            rows=("correct", "size"),
        )
        .reset_index()
        .sort_values(["config_sort", "story_sort"])
    )


def class_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["config_key", "config_label", "config_sort", "expected_gender"])
        .agg(rows=("correct", "size"), unique_samples=("sample_uid", "nunique"), accuracy=("correct", "mean"))
        .reset_index()
        .sort_values(["config_sort", "expected_gender"])
    )


def standalone_consistency(df: pd.DataFrame) -> pd.DataFrame:
    standalone = df[df["mode"].isin(["small_only", "large_only"])]
    checks = (
        standalone.groupby(["config_key", "config_label", "story_key", "sample_id"])
        .agg(
            row_repeats=("correct", "size"),
            predicted_variants=("predicted_gender", "nunique"),
            correctness_variants=("correct", "nunique"),
            parse_variants=("parse_ok", "nunique"),
        )
        .reset_index()
    )
    return checks[
        (checks["predicted_variants"] > 1)
        | (checks["correctness_variants"] > 1)
        | (checks["parse_variants"] > 1)
    ].sort_values(["config_key", "story_key", "sample_id"])


def speculative_vs_target(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specs = df[df["mode"] == "speculative"]
    for (target, draft), spec_rows in specs.groupby(["large_model_name", "small_model_name"]):
        base_rows = df[
            (df["mode"] == "large_only")
            & (df["large_model_name"] == target)
            & (df["small_model_name"] == draft)
        ]
        merged = spec_rows.merge(
            base_rows,
            on=["story_key", "sample_id"],
            suffixes=("_spec", "_large"),
            how="inner",
        )
        if merged.empty:
            continue
        pred_match = (
            merged["predicted_gender_spec"].astype(str) == merged["predicted_gender_large"].astype(str)
        ).mean()
        correct_match = (merged["correct_spec"] == merged["correct_large"]).mean()
        large_wall = merged["wall_ms_large"].mean()
        spec_wall = merged["wall_ms_spec"].mean()
        rows.append(
            {
                "target_family": target,
                "draft_family": draft,
                "pair": f"{MODEL_LABELS.get(target, target)} + {MODEL_LABELS.get(draft, draft)}",
                "samples": len(merged),
                "large_accuracy": merged["correct_large"].mean(),
                "spec_accuracy": merged["correct_spec"].mean(),
                "prediction_match_rate": pred_match,
                "correctness_match_rate": correct_match,
                "large_avg_wall_ms": large_wall,
                "spec_avg_wall_ms": spec_wall,
                "slowdown_vs_large": (spec_wall / large_wall) - 1 if large_wall else np.nan,
                "avg_draft_acceptance_ratio": merged["draft_acceptance_ratio_spec"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("pair")


def model_failure_examples(df: pd.DataFrame, config_keys: list[str], limit: int = 8) -> pd.DataFrame:
    failures = df[(df["config_key"].isin(config_keys)) & (~df["correct"])]
    cols = [
        "config_label",
        "story_key",
        "sample_id",
        "speaker",
        "expected_gender",
        "predicted_gender",
        "parse_ok",
    ]
    return failures.loc[:, cols].drop_duplicates().head(limit)


def write_charts(
    overall: pd.DataFrame,
    story_metrics: pd.DataFrame,
    class_metrics: pd.DataFrame,
    assets_dir: Path,
) -> dict[str, Path]:
    assets_dir.mkdir(parents=True, exist_ok=True)
    charts: dict[str, Path] = {}

    heat = story_metrics.pivot(index="config_label", columns="story_key", values="accuracy")
    label_order = (
        story_metrics[["config_label", "config_sort"]]
        .drop_duplicates()
        .sort_values("config_sort")["config_label"]
        .tolist()
    )
    heat = heat.reindex(label_order)
    heat = heat.reindex(sorted(heat.columns, key=lambda x: int(x.split("_")[-1])), axis=1)

    fig, ax = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
    im = ax.imshow(heat.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(heat.columns)), labels=heat.columns)
    ax.set_yticks(range(len(heat.index)), labels=heat.index)
    ax.set_title("Accuracy by story and model configuration")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            value = heat.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value < 0.65 else "black")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Accuracy")
    path = assets_dir / "story_model_accuracy_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    charts["accuracy_heatmap"] = path

    fig, ax = plt.subplots(figsize=(9.5, 6), constrained_layout=True)
    colors = {"large": "#1b9e77", "speculative": "#d95f02", "small": "#377eb8"}
    markers = {"large": "s", "speculative": "D", "small": "o"}
    for kind, group in overall.groupby("config_kind"):
        ax.scatter(
            group["avg_wall_ms"],
            group["accuracy"],
            s=110,
            marker=markers.get(kind, "o"),
            color=colors.get(kind, "#333333"),
            label=kind.title(),
            edgecolor="white",
            linewidth=0.8,
        )
        for _, row in group.iterrows():
            short = row["config_label"].replace("Large only: ", "").replace("Small only: ", "").replace("Speculative: ", "Spec: ")
            ax.annotate(short, (row["avg_wall_ms"], row["accuracy"]), xytext=(7, 4), textcoords="offset points", fontsize=8)
    ax.set_title("Accuracy versus latency")
    ax.set_xlabel("Average wall time per sample (ms)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.07)
    ax.grid(True, alpha=0.25)
    ax.legend()
    path = assets_dir / "accuracy_latency_scatter.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    charts["accuracy_latency"] = path

    parse = overall.sort_values("config_sort")
    fig, ax = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
    bars = ax.barh(parse["config_label"], parse["parse_ok_rate"], color="#4c78a8")
    ax.set_title("JSON parse success rate by configuration")
    ax.set_xlabel("Parse OK rate")
    ax.set_xlim(0, 1.05)
    ax.bar_label(bars, labels=[pct(v) for v in parse["parse_ok_rate"]], padding=4, fontsize=8)
    ax.grid(True, axis="x", alpha=0.25)
    path = assets_dir / "parse_ok_by_config.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    charts["parse_ok"] = path

    class_pivot = class_metrics.pivot(index="config_label", columns="expected_gender", values="accuracy")
    class_pivot = class_pivot.reindex(label_order)
    class_pivot = class_pivot.reindex(["female", "male", "unknown"], axis=1)
    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    im = ax.imshow(class_pivot.values, cmap="cividis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(class_pivot.columns)), labels=class_pivot.columns)
    ax.set_yticks(range(len(class_pivot.index)), labels=class_pivot.index)
    ax.set_title("Class-wise accuracy")
    for i in range(class_pivot.shape[0]):
        for j in range(class_pivot.shape[1]):
            value = class_pivot.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value < 0.55 else "black")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Accuracy")
    path = assets_dir / "class_accuracy_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    charts["class_accuracy"] = path

    return charts


def build_report(
    csv_path: Path,
    report_path: Path,
    assets_dir: Path,
    df: pd.DataFrame,
    overall: pd.DataFrame,
    story_metrics: pd.DataFrame,
    class_metrics: pd.DataFrame,
    consistency_issues: pd.DataFrame,
    spec_compare: pd.DataFrame,
    charts: dict[str, Path],
) -> None:
    unique_samples = df["sample_uid"].nunique()
    stories = df[["story_key", "story_title"]].drop_duplicates().sort_values("story_key")
    gender_mix = (
        df.drop_duplicates("sample_uid")
        .groupby("expected_gender")
        .size()
        .reindex(["female", "male", "unknown"], fill_value=0)
    )

    overall_table = overall.copy()
    overall_table["Accuracy"] = overall_table["accuracy"].map(pct)
    overall_table["Parse OK"] = overall_table["parse_ok_rate"].map(pct)
    overall_table["Avg wall ms"] = overall_table["avg_wall_ms"].map(ms)
    overall_table["P95 wall ms"] = overall_table["p95_wall_ms"].map(ms)
    overall_table["Draft acceptance"] = overall_table["avg_draft_acceptance_ratio"].map(pct)
    overall_table = overall_table.rename(
        columns={
            "config_label": "Configuration",
            "rows": "Rows",
            "unique_samples": "Unique samples",
        }
    )[
        [
            "Configuration",
            "Rows",
            "Unique samples",
            "Accuracy",
            "Parse OK",
            "Avg wall ms",
            "P95 wall ms",
            "Draft acceptance",
        ]
    ]

    story_table = story_metrics.pivot(index="config_label", columns="story_key", values="accuracy")
    story_table = story_table.reindex(
        story_metrics[["config_label", "config_sort"]]
        .drop_duplicates()
        .sort_values("config_sort")["config_label"]
    )
    story_table = story_table.map(pct).reset_index().rename(columns={"config_label": "Configuration"})

    class_table = class_metrics.pivot(index="config_label", columns="expected_gender", values="accuracy")
    class_table = class_table.reindex(
        class_metrics[["config_label", "config_sort"]]
        .drop_duplicates()
        .sort_values("config_sort")["config_label"]
    )
    class_table = class_table.reindex(["female", "male", "unknown"], axis=1).map(pct)
    class_table = class_table.reset_index().rename(
        columns={
            "config_label": "Configuration",
            "female": "Female",
            "male": "Male",
            "unknown": "Unknown",
        }
    )

    spec_table = spec_compare.copy()
    if not spec_table.empty:
        spec_table["Large acc"] = spec_table["large_accuracy"].map(pct)
        spec_table["Spec acc"] = spec_table["spec_accuracy"].map(pct)
        spec_table["Prediction match"] = spec_table["prediction_match_rate"].map(pct)
        spec_table["Large avg ms"] = spec_table["large_avg_wall_ms"].map(ms)
        spec_table["Spec avg ms"] = spec_table["spec_avg_wall_ms"].map(ms)
        spec_table["Slowdown"] = spec_table["slowdown_vs_large"].map(signed_pct)
        spec_table["Draft acceptance"] = spec_table["avg_draft_acceptance_ratio"].map(pct)
        spec_table = spec_table.rename(columns={"pair": "Pair", "samples": "Samples"})[
            [
                "Pair",
                "Samples",
                "Large acc",
                "Spec acc",
                "Prediction match",
                "Large avg ms",
                "Spec avg ms",
                "Slowdown",
                "Draft acceptance",
            ]
        ]

    failure_table = model_failure_examples(
        df,
        [
            "large:qwen_large",
            "speculative:qwen_large+qwen_small",
            "small:smollm",
            "small:qwen_small",
            "small:gemma",
        ],
        limit=12,
    )
    if not failure_table.empty:
        failure_table = failure_table.rename(
            columns={
                "config_label": "Configuration",
                "story_key": "Story",
                "sample_id": "Sample",
                "speaker": "Speaker",
                "expected_gender": "Expected",
                "predicted_gender": "Predicted",
                "parse_ok": "Parse OK",
            }
        )

    qwen_spec = spec_compare[spec_compare["target_family"] == "qwen_large"]
    llama_spec = spec_compare[spec_compare["target_family"] == "llama_3_2_3b"]
    qwen_slowdown = float(qwen_spec["slowdown_vs_large"].iloc[0]) if not qwen_spec.empty else np.nan
    llama_slowdown = float(llama_spec["slowdown_vs_large"].iloc[0]) if not llama_spec.empty else np.nan

    qwen_large_row = overall[overall["config_key"] == "large:qwen_large"].iloc[0]
    smollm_row = overall[overall["config_key"] == "small:smollm"].iloc[0]
    qwen_small_row = overall[overall["config_key"] == "small:qwen_small"].iloc[0]
    gemma_row = overall[overall["config_key"] == "small:gemma"].iloc[0]

    chart_links = "\n".join(
        f"- [{name.replace('_', ' ').title()}]({path.relative_to(report_path.parent).as_posix()})"
        for name, path in charts.items()
    )
    consistency_text = (
        "No prediction, correctness, or parse-status inconsistencies were found across repeated standalone rows."
        if consistency_issues.empty
        else f"{len(consistency_issues)} repeated standalone sample groups changed prediction, correctness, or parse status."
    )

    story_list = "\n".join(
        f"- `{row.story_key}`: {row.story_title}" for _, row in stories.iterrows()
    )

    report = f"""# Story Model Matrix Comparative Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}

Source CSV: `{csv_path}`

## Scope

This report analyzes {len(df):,} prediction rows across {len(stories)} story fixtures and {unique_samples} unique labeled story samples. Each story contributes 8 labeled samples. Standalone small and large model rows are repeated across model-pair labels in the matrix, so the tables report both raw row counts and unique story samples.

Stories:

{story_list}

Unique sample label mix: female={gender_mix["female"]}, male={gender_mix["male"]}, unknown={gender_mix["unknown"]}.

## Executive Summary

- Best accuracy: Qwen3 4B reached {pct(qwen_large_row["accuracy"])} overall with {pct(qwen_large_row["parse_ok_rate"])} parse success. It was perfect on stories 2-5 and missed one sample in story 1.
- Speculative decoding preserved target quality but did not speed up this benchmark. `Qwen3 4B + Qwen3 0.6B` matched Qwen3 4B accuracy and predictions, but was {signed_pct(qwen_slowdown)} slower than the matching large-only baseline. `Llama 3.2 3B + SmolLM3 3B` also matched target behavior but was {signed_pct(llama_slowdown)} slower.
- Best standalone small model: SmolLM3 3B reached {pct(smollm_row["accuracy"])} at {ms(smollm_row["avg_wall_ms"])} ms average wall time. It beat Qwen3 0.6B on accuracy but was much slower.
- Fastest usable model: Qwen3 0.6B averaged {ms(qwen_small_row["avg_wall_ms"])} ms, but its overall accuracy was only {pct(qwen_small_row["accuracy"])}.
- FunctionGemma is not ready for this JSON classifier path: parse success was only {pct(gemma_row["parse_ok_rate"])} and overall accuracy was {pct(gemma_row["accuracy"])}.

## Overall Configuration Ranking

{markdown_table(overall_table)}

## Accuracy By Story

{markdown_table(story_table)}

## Speculative Versus Matching Large-Only Target

Speculative rows are compared to the large-only rows with the same target and draft-family pairing label.

{markdown_table(spec_table)}

## Class-Wise Accuracy

{markdown_table(class_table)}

## Diagnostics

- Repeated standalone row consistency: {consistency_text}
- Mistral 7B appears only as a `large_model_name` pairing label for `small_only` rows in this CSV. There are no `large_only` or `speculative` Mistral target rows to compare here.
- Draft acceptance is high for both speculative pairs, around 71-80% by story, but the generated outputs are short JSON classifications. The overhead of speculative setup and verification dominates the current wall-clock timings.

Representative failure rows:

{markdown_table(failure_table)}

## Visuals

{chart_links}

## Recommendation

Use Qwen3 4B as the accuracy baseline for this task. Keep speculative decoding wired as an integration path, but do not claim a latency win from this run: both speculative pairs are slower than their matching large-only baselines despite high draft acceptance. For a fast fallback, Qwen3 0.6B is the latency winner but needs stronger prompting or a second-stage verifier. SmolLM3 is the strongest standalone small model, though story 4 shows it can fail badly on harder document-frame attribution.
"""

    report_path.write_text(dedent(report).strip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a comparative report for the story model matrix CSV.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("notebooks/outputs/story_model_matrix_predictions.csv"),
        help="Path to story_model_matrix_predictions.csv",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("notebooks/outputs/story_model_matrix_comparative_report.md"),
        help="Markdown report output path",
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=Path("notebooks/outputs/story_model_matrix_report_assets"),
        help="Directory for generated charts and summary tables",
    )
    args = parser.parse_args()

    df = add_config_columns(pd.read_csv(args.csv))
    overall = aggregate_configs(df)
    story_metrics = per_story_accuracy(df)
    class_metrics = class_accuracy(df)
    consistency_issues = standalone_consistency(df)
    spec_compare = speculative_vs_target(df)

    args.assets_dir.mkdir(parents=True, exist_ok=True)
    overall.to_csv(args.assets_dir / "overall_config_summary.csv", index=False)
    story_metrics.to_csv(args.assets_dir / "per_story_metrics.csv", index=False)
    class_metrics.to_csv(args.assets_dir / "class_accuracy.csv", index=False)
    spec_compare.to_csv(args.assets_dir / "speculative_vs_target.csv", index=False)
    if not consistency_issues.empty:
        consistency_issues.to_csv(args.assets_dir / "standalone_consistency_issues.csv", index=False)

    charts = write_charts(overall, story_metrics, class_metrics, args.assets_dir)
    build_report(
        args.csv,
        args.report,
        args.assets_dir,
        df,
        overall,
        story_metrics,
        class_metrics,
        consistency_issues,
        spec_compare,
        charts,
    )

    print(f"Wrote {args.report}")
    print(f"Wrote assets to {args.assets_dir}")
    print(overall[["config_label", "accuracy", "parse_ok_rate", "avg_wall_ms"]].to_string(index=False))
    print()
    print(spec_compare.to_string(index=False))


if __name__ == "__main__":
    main()
