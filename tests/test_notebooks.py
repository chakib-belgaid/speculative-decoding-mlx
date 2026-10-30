import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_NOTEBOOK = PROJECT_ROOT / "notebooks" / "benchmark_analysis.ipynb"
SPECULATIVE_WALKTHROUGH_NOTEBOOK = PROJECT_ROOT / "notebooks" / "speculative_decoding_walkthrough.ipynb"


def test_benchmark_analysis_notebook_is_dedicated_to_benchmarks_without_pdf_reader() -> None:
    notebook = json.loads(BENCHMARK_NOTEBOOK.read_text(encoding="utf-8"))
    serialized_notebook = json.dumps(notebook)
    sources = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
    )

    assert "# Benchmark Analysis" in sources
    assert "load_story_benchmark_dataframe" in sources
    assert "summarize_results_dataframe" in sources
    assert "pdf" not in sources.lower()
    assert ".local_path" not in sources
    assert "str(spec.path) if spec.path else" in sources
    assert "AttributeError" not in serialized_notebook


def test_speculative_walkthrough_has_story_model_matrix_with_fixed_paper_gamma() -> None:
    notebook = json.loads(SPECULATIVE_WALKTHROUGH_NOTEBOOK.read_text(encoding="utf-8"))
    sources = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
    )

    assert "PAPER_DEFAULT_GAMMA = 7" in sources
    assert "STORY_MODEL_MATRIX_CSV" in sources
    assert "summarize_story_model_matrix_dataframe" in sources
    assert 'STORY_MATRIX_HEATMAP_MODES = ["small_only", "large_only", "speculative"]' in sources
    assert "for heatmap_mode in STORY_MATRIX_HEATMAP_MODES" in sources
    assert "GAMMA_GRID" not in sources
    assert "for gamma in" not in sources
