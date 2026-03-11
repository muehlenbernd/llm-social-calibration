"""
Correlation analysis script: computes human–LLM concordance and residual plots.

Usage:
    python -m src.analysis.main_correl
"""

import pandas as pd

pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)

from src.analysis.metrics import (
    HUMAN_MAIN_EFFECTS,
    HUMAN_INTERACTIONS,
    ATTRIBUTES,
    SIGNIFICANT_MAIN_ATTRS,
    SIGNIFICANT_INTERACTION_ATTRS,
    load_human_data,
    load_llm_data,
    compute_hlc_modelwise,
    compute_comprehensive_alignment_metrics,
)

from src.analysis.metrics_plotter import (
    plot_hlc_grid,
    plot_hlc_residuals_grid,
)

import numpy as np

# ── Table formatting utilities ─────────────────────────────────────────────────

def format_enhanced_table(df, cols, higher_is_better=None, lower_is_better=None):
    """
    Formats a table with model separators and highlights best values per model.

    Args:
        df: DataFrame to format
        cols: columns to include in the output
        higher_is_better: list of columns where higher values are better
        lower_is_better: list of columns where lower values are better
    """
    if higher_is_better is None:
        higher_is_better = []
    if lower_is_better is None:
        lower_is_better = []

    bias_metrics = {"systematic_bias": 0, "proportional_bias": 1}

    output_lines = []
    current_model = None

    for idx, row in df.iterrows():
        model = row.get("model", "")
        if model != current_model:
            if current_model is not None:
                output_lines.append("-" * 80)
            current_model = model

        row_str = " | ".join(
            f"{col}: {row[col]:.3f}" if isinstance(row[col], float) else f"{col}: {row[col]}"
            for col in cols if col in row
        )
        output_lines.append(row_str)

    return "\n".join(output_lines)


# ── Paths ──────────────────────────────────────────────────────────────────────

HUMAN_PATH = "data/human_ratings/impX1.csv"

LLM_RESULTS = {
    "MIN": "data/llm_ratings/results_MIN.json",
    "ALT": "data/llm_ratings/results_ALT.json",
    "KMA": "data/llm_ratings/results_KMA.json",
    "COM": "data/llm_ratings/results_COM.json",
}

# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    human_df = load_human_data(HUMAN_PATH)

    for prompt_label, llm_path in LLM_RESULTS.items():
        print(f"\n{'='*60}")
        print(f"Prompt condition: {prompt_label}")
        print(f"{'='*60}")

        llm_df = load_llm_data(llm_path)

        alignment = compute_hlc_modelwise(human_df, llm_df)
        print(alignment.to_string())

        comprehensive = compute_comprehensive_alignment_metrics(human_df, llm_df)
        print(format_enhanced_table(
            comprehensive,
            cols=["model", "spearman_rho", "ccc", "rmse", "systematic_bias", "proportional_bias"],
            higher_is_better=["spearman_rho", "ccc"],
            lower_is_better=["rmse"],
        ))
