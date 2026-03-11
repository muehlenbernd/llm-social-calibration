# analysis_metrics.py

import json
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns


# ==================================================
# 1. Canonical mappings (EXPLICIT AND VERIFIED)
# ==================================================

SCENARIO_MAP_HUMAN_TO_CANON = {
    "house": "house",
    "conference": "conference",
    "cinema": "cinema",
    "bike": "bicycle",
    "cooking": "pasta",
    "absence": "office"
}

CONTEXT_MAP_HUMAN = {
    "lowPr": "low",
    "highPr": "high"
}

ATTRIBUTE_MAP_HUMAN = {
    "Acomp": "competent",
    "Aknowl": "knowledgeable",
    "Awellp": "well-prepared",
    "Ahelp": "helpful",
    "Alike": "likeable",
    "Apednt": "pedantic"
}

ATTRIBUTES = list(ATTRIBUTE_MAP_HUMAN.values())


# ==================================================
# Human effect ground truth (from Experiment 1)
# ==================================================

# Main effects of FORM (precise vs approx)
HUMAN_MAIN_EFFECTS = {
    "competent": True,
    "knowledgeable": True,
    "well-prepared": True,
    "helpful": True,
    "likeable": False,   # no significant main effect
    "pedantic": True
}

# FORM × CONTEXT interactions
HUMAN_INTERACTIONS = {
    "competent": True,
    "knowledgeable": True,
    "well-prepared": True,
    "helpful": True,
    "likeable": True,
    "pedantic": False    # no significant interaction
}

SIGNIFICANT_MAIN_ATTRS = {
    "competent",
    "knowledgeable",
    "well-prepared",
    "helpful",
    "pedantic"
}

SIGNIFICANT_INTERACTION_ATTRS = {
    "competent",
    "knowledgeable",
    "well-prepared",
    "helpful",
    "likeable"
}

# ==================================================
# 2. Sanity-check utilities
# ==================================================

def assert_no_missing(series, name):
    if series.isna().any():
        missing = series[series.isna()]
        raise ValueError(
            f"[SANITY CHECK FAILED] Missing mappings in {name}: "
            f"{missing.unique()}"
        )


# ==================================================
# 3. Load + normalize HUMAN data
# ==================================================

def load_human_data(path: str) -> pd.DataFrame:
    """
    Returns long-format human data:
    scenario | context | utterance | attribute | rating
    """
    df = pd.read_csv(path)

    # --- scenario ---
    df["scenario"] = df["Xscenario"].map(SCENARIO_MAP_HUMAN_TO_CANON)
    assert_no_missing(df["scenario"], "human scenario")

    # --- context ---
    df["context"] = df["Xcontext"].map(CONTEXT_MAP_HUMAN)
    assert_no_missing(df["context"], "human context")

    # --- utterance ---
    df["utterance"] = df["Xanswer"]
    if not set(df["utterance"].unique()) <= {"precise", "approx"}:
        raise ValueError("[SANITY CHECK FAILED] Unexpected utterance labels")

    # --- ratings: wide → long ---
    rating_cols = list(ATTRIBUTE_MAP_HUMAN.keys())

    long_df = df.melt(
        id_vars=["scenario", "context", "utterance"],
        value_vars=rating_cols,
        var_name="attribute_raw",
        value_name="rating"
    )

    long_df["attribute"] = long_df["attribute_raw"].map(ATTRIBUTE_MAP_HUMAN)
    assert_no_missing(long_df["attribute"], "human attribute")

    return long_df[["scenario", "context", "utterance", "attribute", "rating"]]


# ==================================================
# 4. Load + normalize LLM data
# ==================================================

def load_llm_data(path: str) -> pd.DataFrame:
    """
    Returns normalized LLM data:
    model | scenario | context | utterance | attribute | rating
    """
    with open(path, "r") as f:
        raw = json.load(f)

    df = pd.DataFrame(raw)
    df = df[df["valid"] & df["likert"].notna()]

    # --- sanity checks ---
    if not set(df["utterance"].unique()) <= {"precise", "approx"}:
        raise ValueError("[SANITY CHECK FAILED] Unexpected LLM utterance labels")

    if not set(df["context"].unique()) <= {"low", "high"}:
        raise ValueError("[SANITY CHECK FAILED] Unexpected LLM context labels")

    if not set(df["attribute"].unique()) <= set(ATTRIBUTES):
        raise ValueError("[SANITY CHECK FAILED] Unexpected LLM attribute labels")

    return df.rename(columns={"likert": "rating"})[
        ["model", "scenario", "context", "utterance", "attribute", "rating"]
    ]


# ==================================================
# 5. Aggregation helpers
# ==================================================

def mean_by(df, cols):
    return df.groupby(cols)["rating"].mean().reset_index()


def delta_precise_minus_approx(df):
    """
    Expects utterance ∈ {precise, approx}
    """
    p = df[df["utterance"] == "precise"]["rating"].mean()
    a = df[df["utterance"] == "approx"]["rating"].mean()
    return p - a


# ==================================================
# 6. Directional Agreement Score (global)
# ==================================================

def compute_das(human_df, llm_df):
    results = {}

    for attr in ATTRIBUTES:
        h = human_df[human_df["attribute"] == attr]
        l = llm_df[llm_df["attribute"] == attr]

        h_delta = delta_precise_minus_approx(h)
        l_delta = delta_precise_minus_approx(l)

        if not HUMAN_MAIN_EFFECTS[attr]:
            agreement = np.nan  # not defined
        else:
            agreement = int(np.sign(h_delta) == np.sign(l_delta))

        results[attr] = {
            "human_delta": h_delta,
            "llm_delta": l_delta,
            "agreement": agreement
        }

    return results



# ==================================================
# 7. Scenario-wise DAS
# ==================================================

def compute_scenario_das(human_df, llm_df):
    records = []

    for attr in ATTRIBUTES:
        h_attr = human_df[human_df["attribute"] == attr]

        # human reference sign (aggregated)
        ref_sign = np.sign(delta_precise_minus_approx(h_attr))

        for scenario in h_attr["scenario"].unique():
            h_s = h_attr[h_attr["scenario"] == scenario]
            l_s = llm_df[
                (llm_df["attribute"] == attr) &
                (llm_df["scenario"] == scenario)
            ]

            if len(l_s) == 0:
                continue

            l_delta = delta_precise_minus_approx(l_s)

            records.append({
                "attribute": attr,
                "scenario": scenario,
                "llm_delta": l_delta,
                "agreement": int(np.sign(l_delta) == ref_sign)
            })

    return pd.DataFrame(records)


# ==================================================
# 8. Interaction Sensitivity Score (global)
# ==================================================

def compute_iss(human_df, llm_df):
    results = {}

    for attr in ATTRIBUTES:
        h = human_df[human_df["attribute"] == attr]
        l = llm_df[llm_df["attribute"] == attr]

        def interaction(df):
            high = delta_precise_minus_approx(df[df["context"] == "high"])
            low = delta_precise_minus_approx(df[df["context"] == "low"])
            return high - low

        h_dd = interaction(h)
        l_dd = interaction(l)

        if not HUMAN_INTERACTIONS[attr]:
            agreement = np.nan  # not defined
        else:
            agreement = int(np.sign(h_dd) == np.sign(l_dd))

        results[attr] = {
            "human_interaction": h_dd,
            "llm_interaction": l_dd,
            "agreement": agreement
        }

    return results



# ==================================================
# 9. Scenario-wise ISS
# ==================================================

def compute_scenario_iss(human_df, llm_df):
    records = []

    for attr in ATTRIBUTES:
        h_attr = human_df[human_df["attribute"] == attr]

        ref_dd = (
            delta_precise_minus_approx(h_attr[h_attr["context"] == "high"]) -
            delta_precise_minus_approx(h_attr[h_attr["context"] == "low"])
        )
        ref_sign = np.sign(ref_dd)

        for scenario in h_attr["scenario"].unique():
            l_s = llm_df[
                (llm_df["attribute"] == attr) &
                (llm_df["scenario"] == scenario)
            ]

            if len(l_s) == 0:
                continue

            dd = (
                delta_precise_minus_approx(l_s[l_s["context"] == "high"]) -
                delta_precise_minus_approx(l_s[l_s["context"] == "low"])
            )

            records.append({
                "attribute": attr,
                "scenario": scenario,
                "llm_interaction": dd,
                "agreement": int(np.sign(dd) == ref_sign)
            })

    return pd.DataFrame(records)


# ==================================================
# 10. Human–LLM Correlation
# ==================================================

def compute_hlc(human_df, llm_df):
    h_means = mean_by(
        human_df, ["attribute", "context", "utterance"]
    )
    l_means = mean_by(
        llm_df, ["attribute", "context", "utterance"]
    )

    merged = pd.merge(
        h_means, l_means,
        on=["attribute", "context", "utterance"],
        suffixes=("_human", "_llm")
    )

    rho, p = spearmanr(
        merged["rating_human"],
        merged["rating_llm"]
    )

    return rho, p

# ==================================================
# 11. Human–LLM Correlation (model-wise)
# ==================================================

def compute_hlc_modelwise(human_df, llm_df, scenario_wise=False):
    """
    Computes Spearman correlation between human and LLM mean ratings
    across attribute × context × utterance [× scenario].

    Args:
        human_df: DataFrame with human ratings
        llm_df: DataFrame with LLM ratings
        scenario_wise: If True, include scenario in grouping (144 conditions)
                      If False, aggregate across scenarios (24 conditions)

    Returns:
        rho (float), p_value (float)
    """

    # Choose grouping columns based on scenario_wise flag
    group_cols = ["attribute", "context", "utterance"]
    if scenario_wise:
        group_cols.append("scenario")

    # aggregate human means
    human_means = (
        human_df
        .groupby(group_cols)["rating"]
        .mean()
        .reset_index()
        .rename(columns={"rating": "rating_human"})
    )

    # aggregate LLM means
    llm_means = (
        llm_df
        .groupby(group_cols)["rating"]
        .mean()
        .reset_index()
        .rename(columns={"rating": "rating_llm"})
    )

    merged = pd.merge(
        human_means,
        llm_means,
        on=group_cols,
        how="inner"
    )

    # sanity check
    if len(merged) < 2:
        raise ValueError(
            f"HLC undefined: only {len(merged)} aligned conditions"
        )

    rho, p = spearmanr(
        merged["rating_human"],
        merged["rating_llm"]
    )

    return rho, p


def compute_comprehensive_alignment_metrics(human_df, llm_df, scenario_wise=False):
    """
    Computes comprehensive alignment metrics between human and LLM ratings.
    
    Args:
        human_df: DataFrame with human ratings
        llm_df: DataFrame with LLM ratings  
        scenario_wise: If True, include scenario in grouping (144 conditions)
                      If False, aggregate across scenarios (24 conditions)
    
    Returns:
        dict: Dictionary containing multiple alignment measures
    """
    from scipy.stats import pearsonr
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import numpy as np
    
    # Choose grouping columns based on scenario_wise flag
    group_cols = ["attribute", "context", "utterance"]
    if scenario_wise:
        group_cols.append("scenario")
    
    # aggregate human means
    human_means = (
        human_df
        .groupby(group_cols)["rating"]
        .mean()
        .reset_index()
        .rename(columns={"rating": "rating_human"})
    )

    # aggregate LLM means
    llm_means = (
        llm_df
        .groupby(group_cols)["rating"]
        .mean()
        .reset_index()
        .rename(columns={"rating": "rating_llm"})
    )

    merged = pd.merge(
        human_means,
        llm_means,
        on=group_cols,
        how="inner"
    )

    if len(merged) < 2:
        raise ValueError("Not enough aligned conditions for meaningful analysis")

    h = merged["rating_human"].values
    m = merged["rating_llm"].values
    
    # Basic correlations
    spearman_rho, spearman_p = spearmanr(h, m)
    pearson_r, pearson_p = pearsonr(h, m)
    
    # Error metrics
    mse = mean_squared_error(h, m)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(h, m)
    
    # Concordance Correlation Coefficient (CCC)
    h_mean, m_mean = np.mean(h), np.mean(m)
    h_var, m_var = np.var(h, ddof=1), np.var(m, ddof=1)
    covariance = np.mean((h - h_mean) * (m - m_mean))
    ccc = (2 * covariance) / (h_var + m_var + (h_mean - m_mean)**2)
    
    # Systematic vs random error components
    systematic_bias = m_mean - h_mean
    proportional_bias = m_var / h_var if h_var > 0 else np.nan
    
    # R-squared from regression
    slope, intercept = np.polyfit(h, m, 1)
    predicted = slope * h + intercept
    ss_res = np.sum((m - predicted) ** 2)
    ss_tot = np.sum((m - m_mean) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "ccc": ccc,
        "systematic_bias": systematic_bias,
        "proportional_bias": proportional_bias,
        "r_squared": r_squared,
        "n_conditions": len(merged)
    }

# ==================================================
# 12. Effect Size Ratio (ESR)
# ==================================================

def compute_esr(human_df, llm_df):
    """
    Computes ESR per attribute.
    Returns dict:
        attr -> {
            human_delta,
            llm_delta,
            esr
        }
    """

    results = {}

    for attr in ATTRIBUTES:
        # skip attributes with no human main effect
        if not HUMAN_MAIN_EFFECTS.get(attr, False):
            results[attr] = {
                "human_delta": None,
                "llm_delta": None,
                "esr": None
            }
            continue

        h = human_df[human_df["attribute"] == attr]
        l = llm_df[llm_df["attribute"] == attr]

        human_delta = delta_precise_minus_approx(h)
        llm_delta = delta_precise_minus_approx(l)

        # guard against division by ~0
        if abs(human_delta) < 1e-6:
            esr = None
        else:
            esr = abs(llm_delta) / abs(human_delta)

        results[attr] = {
            "human_delta": human_delta,
            "llm_delta": llm_delta,
            "esr": esr
        }

    return results


# ==================================================
# 13. Interaction Effect Size Ratio (ESR_interaction)
# ==================================================

def compute_esr_interaction(human_df, llm_df):
    """
    Computes interaction-level ESR per attribute.

    Interaction = (precise - approx)_high - (precise - approx)_low

    Returns dict:
        attr -> {
            human_interaction,
            llm_interaction,
            esr_interaction
        }
    """

    results = {}

    for attr in ATTRIBUTES:
        # skip attributes with no human interaction
        if not HUMAN_INTERACTIONS.get(attr, False):
            results[attr] = {
                "human_interaction": None,
                "llm_interaction": None,
                "esr_interaction": None
            }
            continue

        h = human_df[human_df["attribute"] == attr]
        l = llm_df[llm_df["attribute"] == attr]

        def interaction_delta(df):
            high = delta_precise_minus_approx(
                df[df["context"] == "high"]
            )
            low = delta_precise_minus_approx(
                df[df["context"] == "low"]
            )
            return high - low

        human_dd = interaction_delta(h)
        llm_dd = interaction_delta(l)

        # guard against division by ~0
        if abs(human_dd) < 1e-6:
            esr_int = None
        else:
            esr_int = abs(llm_dd) / abs(human_dd)

        results[attr] = {
            "human_interaction": human_dd,
            "llm_interaction": llm_dd,
            "esr_interaction": esr_int
        }

    return results

# ==================================================
# ESR (main effects) – refactored, all attributes
# ==================================================

def compute_esr_main_all(human_df, llm_df):
    """
    Computes main-effect ESR for ALL attributes.

    Returns dict:
      attr -> {
        human_delta,
        llm_delta,
        esr_main
      }
    """

    results = {}

    for attr in ATTRIBUTES:
        h = human_df[human_df["attribute"] == attr]
        l = llm_df[llm_df["attribute"] == attr]

        human_delta = delta_precise_minus_approx(h)
        llm_delta = delta_precise_minus_approx(l)

        if abs(human_delta) < 1e-6:
            esr = float("inf") if abs(llm_delta) > 1e-6 else 0.0
        else:
            esr = abs(llm_delta) / abs(human_delta)

        results[attr] = {
            "human_effect": human_delta,
            "llm_effect": llm_delta,
            "esr_main": esr
        }

    return results

# ==================================================
# ESR (main effects) – refactored, all attributes
# ==================================================

def compute_esr_main_all(human_df, llm_df):
    """
    Computes main-effect ESR for ALL attributes.

    Returns dict:
      attr -> {
        human_delta,
        llm_delta,
        esr_main
      }
    """

    results = {}

    for attr in ATTRIBUTES:
        h = human_df[human_df["attribute"] == attr]
        l = llm_df[llm_df["attribute"] == attr]

        human_delta = delta_precise_minus_approx(h)
        llm_delta = delta_precise_minus_approx(l)

        if abs(human_delta) < 1e-6:
            esr = float("inf") if abs(llm_delta) > 1e-6 else 0.0
        else:
            esr = abs(llm_delta) / abs(human_delta)

        results[attr] = {
            "human_effect": human_delta,
            "llm_effect": llm_delta,
            "esr_main": esr
        }

    return results

# ==================================================
# ESR (interaction effects) – refactored, all attributes
# ==================================================

def compute_esr_interaction_all(human_df, llm_df):
    """
    Computes interaction ESR for ALL attributes.

    Interaction = (precise - approx)_high - (precise - approx)_low

    Returns dict:
      attr -> {
        human_interaction,
        llm_interaction,
        esr_interaction
      }
    """

    def interaction_delta(df):
        high = delta_precise_minus_approx(df[df["context"] == "high"])
        low = delta_precise_minus_approx(df[df["context"] == "low"])
        return high - low

    results = {}

    for attr in ATTRIBUTES:
        h = human_df[human_df["attribute"] == attr]
        l = llm_df[llm_df["attribute"] == attr]

        human_dd = interaction_delta(h)
        llm_dd = interaction_delta(l)

        if abs(human_dd) < 1e-6:
            esr = float("inf") if abs(llm_dd) > 1e-6 else 0.0
        else:
            esr = abs(llm_dd) / abs(human_dd)

        results[attr] = {
            "human_effect": human_dd,
            "llm_effect": llm_dd,
            "esr_interaction": esr
        }

    return results


import pandas as pd
import numpy as np

# ==================================================
# Calibration Deviation Score (CDS)
# ==================================================

def compute_CDS_data(
    llm_json_files,
    human_df,
    print_table=False
):
    """
    Computes Calibration Deviation Scores (CDS) for:
        - main effects
        - interaction effects

    Parameters
    ----------
    llm_json_files : list of tuples
        List of (prompt_name, json_path)

        Example:
        [
            ("minimal", "results_minimal_prompt.json"),
            ("uncertainty", "results_uncertainty_prompt.json")
        ]

    human_df : pandas.DataFrame
        Human data (already harmonized)

    print_table : bool
        If True, prints summary table including HLC

    Returns
    -------
    pandas.DataFrame with columns:
        model, prompt, CDS_main, CDS_interaction, HLC
    """

    results = []

    for prompt_name, json_path in llm_json_files:

        llm_df = load_llm_data(json_path)

        for model in llm_df["model"].unique():

            llm_model_df = llm_df[llm_df["model"] == model]

            # --------------------------------------------------
            # Compute ESR dictionaries (existing functions)
            # --------------------------------------------------
            esr_main = compute_esr_main_all(human_df, llm_model_df)
            esr_int  = compute_esr_interaction_all(human_df, llm_model_df)

            # --------------------------------------------------
            # Compute CDS (only attributes with human effect != 0)
            # --------------------------------------------------

            main_devs = []
            for attr, vals in esr_main.items():
                # Only include significant main effect attributes
                if attr not in SIGNIFICANT_MAIN_ATTRS:
                    continue
                    
                human_effect = vals["human_effect"]
                esr = vals["esr_main"]

                if abs(human_effect) > 1e-6 and np.isfinite(esr):
                    main_devs.append(abs(esr - 1))

            int_devs = []
            for attr, vals in esr_int.items():
                # Only include significant interaction effect attributes
                if attr not in SIGNIFICANT_INTERACTION_ATTRS:
                    continue
                    
                human_effect = vals["human_effect"]
                esr = vals["esr_interaction"]

                if abs(human_effect) > 1e-6 and np.isfinite(esr):
                    int_devs.append(abs(esr - 1))

            CDS_main = np.mean(main_devs) if main_devs else np.nan
            CDS_int  = np.mean(int_devs) if int_devs else np.nan

            # --------------------------------------------------
            # Compute HLC (existing function)
            # --------------------------------------------------
            rho, p = compute_hlc(human_df, llm_model_df)
            HLC = rho

            results.append({
                "model": model,
                "prompt": prompt_name,
                "CDS_main": CDS_main,
                "CDS_interaction": CDS_int,
                "HLC": HLC
            })

    results_df = pd.DataFrame(results)

    # --------------------------------------------------
    # Optional pretty printing
    # --------------------------------------------------
    if print_table:
        print("\n" + "=" * 80)
        print("CALIBRATION DEVIATION SUMMARY")
        print("(Lower CDS = better calibration; HLC = structural alignment)")
        print("=" * 80)

        # Define desired prompt order
        prompt_order = ["minimal", "example", "uncertainty"]
        
        # Group by model and print each model's results separately
        for model in results_df["model"].unique():
            model_data = results_df[results_df["model"] == model]
            
            # Sort by custom prompt order
            model_data["prompt_order"] = model_data["prompt"].map(
                {prompt: i for i, prompt in enumerate(prompt_order)}
            )
            model_data = model_data.sort_values("prompt_order").drop("prompt_order", axis=1).round(3)
            
            print(f"\nModel: {model}")
            print("-" * 60)
            print(model_data.to_string(index=False))
            print("-" * 60)

    return results_df


# ==================================================
# 14. CDS Bar Plot Visualization
# ==================================================

def plot_cds_results(results_df, figsize_per_model=(12, 4), save_path=None):
    """
    Creates bar plots of CDS results with multiple rows (one per model).
    
    For each model row, shows:
    - n bars for CDS_main (one per prompt)
    - n bars for CDS_interaction (one per prompt) 
    - n bars for HLC (one per prompt)
    
    Parameters
    ----------
    results_df : pandas.DataFrame
        Output from compute_CDS_data() with columns:
        [model, prompt, CDS_main, CDS_interaction, HLC]
    figsize_per_model : tuple
        (width, height) for each model subplot
    save_path : str, optional
        If provided, saves the plot to this path
    """
    
    # Define prompt order and colors
    prompt_order = ["minimal", "example", "uncertainty"]
    prompt_colors = {
        "minimal": "#1f77b4",      # blue
        "example": "#ff7f0e",      # orange  
        "uncertainty": "#2ca02c"   # green
    }
    
    models = sorted(results_df["model"].unique())
    n_models = len(models)
    
    # Create subplots: one row per model
    fig, axes = plt.subplots(
        nrows=n_models, 
        ncols=1, 
        figsize=(figsize_per_model[0], figsize_per_model[1] * n_models),
        squeeze=False
    )
    
    for i, model in enumerate(models):
        ax = axes[i, 0]
        model_data = results_df[results_df["model"] == model].copy()
        
        # Ensure prompt order
        model_data["prompt_order"] = model_data["prompt"].map(
            {prompt: j for j, prompt in enumerate(prompt_order)}
        )
        model_data = model_data.sort_values("prompt_order")
        
        # Get available prompts for this model
        available_prompts = model_data["prompt"].tolist()
        n_prompts = len(available_prompts)
        
        if n_prompts == 0:
            ax.text(0.5, 0.5, f"No data for {model}", 
                   ha='center', va='center', transform=ax.transAxes)
            continue
            
        # Set up bar positions
        bar_width = 0.25
        x_positions = np.arange(3)  # 3 groups: CDS_main, CDS_interaction, HLC
        
        # Plot bars for each prompt
        for j, prompt in enumerate(available_prompts):
            prompt_data = model_data[model_data["prompt"] == prompt].iloc[0]
            color = prompt_colors.get(prompt, "gray")
            
            # Get values for the three metrics
            values = [
                prompt_data["CDS_main"],
                prompt_data["CDS_interaction"], 
                prompt_data["HLC"]
            ]
            
            # Plot bars with offset for each prompt
            offset = (j - n_prompts/2 + 0.5) * bar_width
            bars = ax.bar(
                x_positions + offset, 
                values, 
                width=bar_width,
                label=prompt,
                color=color,
                alpha=0.8,
                edgecolor='black',
                linewidth=0.5
            )
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                if not np.isnan(val):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width()/2., 
                        height + 0.01,
                        f'{val:.3f}',
                        ha='center', va='bottom',
                        fontsize=8
                    )
        
        # Customize subplot
        ax.set_title(f"Model: {model}", fontsize=14, fontweight='bold')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(["CDS_main", "CDS_interaction", "HLC"])
        ax.set_ylabel("Score")
        ax.grid(axis='y', alpha=0.3)
        
        # Add legend only to first subplot
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Set y-axis limits for better visualization
        ax.set_ylim(bottom=0)
        
        # Add horizontal line at y=1 for CDS metrics (perfect calibration)
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    return fig
