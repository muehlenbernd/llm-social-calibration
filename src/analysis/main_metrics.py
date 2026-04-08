"""
Main analysis script: computes and visualizes evaluation metrics (ESR, CDS, CCC, RMSE, Spearman).

Reproduces Tables 1 and 2 and Figure 1 from the paper.

Usage:
    python -m src.analysis.main_metrics
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
    compute_das,
    compute_iss,
    compute_hlc_modelwise,
    compute_esr,
    compute_esr_interaction,
    compute_esr_main_all,
    compute_esr_interaction_all,
    compute_CDS_data,
    plot_cds_results,
)

from src.analysis.metrics_plotter import (
    plot_hlc_grid,
    plot_hlc_residuals_grid,
    plot_esr_vs_human_strength,
    plot_esr_heatmap,
)

# ── Paths ──────────────────────────────────────────────────────────────────────

HUMAN_PATH = "data/human_ratings/experiment1.csv"

LLM_RESULTS = {
    "MIN": "data/llm_ratings/results_MIN.json",
    "ALT": "data/llm_ratings/results_ALT.json",
    "KMA": "data/llm_ratings/results_KMA.json",
    "COM": "data/llm_ratings/results_COM.json",
}

PLOT_DATA = True

# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    human_df = load_human_data(HUMAN_PATH)

    for prompt_label, llm_path in LLM_RESULTS.items():
        print(f"\n{'='*60}")
        print(f"Prompt condition: {prompt_label}")
        print(f"{'='*60}")

        llm_df = load_llm_data(llm_path)

        # Global pattern similarity (Table 1)
        alignment = compute_hlc_modelwise(human_df, llm_df)
        print("\n--- Global pattern similarity (Spearman ρ, CCC, RMSE) ---")
        print(alignment.to_string())

        # Structural alignment (DAS, ISS)
        das = compute_das(human_df, llm_df)
        iss = compute_iss(human_df, llm_df)
        print("\n--- Structural alignment (DAS, ISS) ---")
        print(f"DAS: {das}")
        print(f"ISS: {iss}")

        # Magnitude calibration (ESR, CDS — Table 2)
        cds_data = compute_CDS_data(human_df, llm_df)
        print("\n--- Calibration Deviation Scores (CDS) ---")
        print(cds_data.to_string())

        # Figure 1: ESR heatmap
        if PLOT_DATA:
            esr_main = compute_esr_main_all(human_df, llm_df)
            esr_inter = compute_esr_interaction_all(human_df, llm_df)
            plot_esr_heatmap(
                esr_main, esr_inter,
                title=f"ESR Heatmap — {prompt_label}",
                save_path=f"results/figures/esr_heatmap_{prompt_label}.png",
            )
