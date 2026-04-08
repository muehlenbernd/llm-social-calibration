# metrics_plotter.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# Effect-type metadata (ground truth)
# --------------------------------------------------

MAIN_EFFECT_ATTRS = {
    "competent",
    "knowledgeable",
    "well-prepared",
    "helpful",
    "pedantic"
}

INTERACTION_ATTRS = {
    "competent",
    "knowledgeable",
    "well-prepared",
    "helpful",
    "likeable"
}


# --------------------------------------------------
# Prepare merged dataframe with effect type
# --------------------------------------------------

# --------------------------------------------------
# Helper: prepare merged human–LLM means
# --------------------------------------------------

def prepare_hlc_dataframe(human_df, llm_df):
    """
    Returns:
    attribute | context | utterance | rating_human | rating_llm
    """

    human_means = (
        human_df
        .groupby(["attribute", "context", "utterance"])["rating"]
        .mean()
        .reset_index()
        .rename(columns={"rating": "rating_human"})
    )

    llm_means = (
        llm_df
        .groupby(["attribute", "context", "utterance"])["rating"]
        .mean()
        .reset_index()
        .rename(columns={"rating": "rating_llm"})
    )

    return pd.merge(
        human_means,
        llm_means,
        on=["attribute", "context", "utterance"],
        how="inner"
    )


# --------------------------------------------------
# Plot 1: 2x3 HLC scatter grid (color = attribute)
# --------------------------------------------------

def plot_hlc_grid(human_df, llm_data_by_prompt, prompts=None, models=None):
    """
    Grid layout with dynamic size:
      rows    = prompt 
      columns = model
    
    Parameters:
    -----------
    human_df : pd.DataFrame
        Human ratings data
    llm_data_by_prompt : dict
        LLM data nested by prompt and model
    prompts : list, optional
        List of prompt names. If None, uses ["minimal", "example"]
    models : list, optional
        List of model names. If None, uses ["claude", "gemini", "gpt"]
    """

    if prompts is None:
        prompts = ["minimal", "example"]
    if models is None:
        models = ["claude", "gemini", "gpt"]

    fig, axes = plt.subplots(
        nrows=len(prompts), ncols=len(models),
        figsize=(5 * len(models), 5 * len(prompts)),
        sharex=True, sharey=True
    )

    # Handle case where axes is 1D or 0D
    if len(prompts) == 1 and len(models) == 1:
        axes = [[axes]]
    elif len(prompts) == 1:
        axes = [axes]
    elif len(models) == 1:
        axes = [[ax] for ax in axes]

    attributes = sorted(human_df["attribute"].unique())
    colors = dict(zip(attributes, plt.cm.tab10.colors))

    for row, prompt in enumerate(prompts):
        for col, model in enumerate(models):
            ax = axes[row][col]
            llm_df = llm_data_by_prompt[prompt][model]

            df = prepare_hlc_dataframe(human_df, llm_df)

            for attr in attributes:
                sub = df[df["attribute"] == attr]
                ax.scatter(
                    sub["rating_human"],
                    sub["rating_llm"],
                    color=colors[attr],
                    alpha=0.75,
                    label=attr if (row == 0 and col == 0) else None
                )

            # diagonal
            ax.plot([1, 7], [1, 7], linestyle="--", linewidth=1)

            ax.set_title(f"{model} – {prompt}")
            ax.set_xlim(1, 7)
            ax.set_ylim(1, 7)

            if col == 0:
                ax.set_ylabel("LLM mean")
            if row == len(prompts) - 1:
                ax.set_xlabel("Human mean")

    # legend
    fig.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=colors[a], label=a, markersize=8)
            for a in attributes
        ],
        loc="lower center",
        ncol=min(len(attributes), len(models)),
        frameon=False
    )

    fig.suptitle(
        "Human–LLM Correlation (HLC)\nColor = Attribute",
        fontsize=16
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.show()

# --------------------------------------------------
# Plot 2: Residuals (LLM − Human), 1x3 grid
# --------------------------------------------------

def plot_hlc_residuals_grid(human_df, llm_data_by_prompt, prompt="example", models=None):
    """
    1xN grid:
      columns = model
      residual = LLM − Human
      color    = attribute
    
    Parameters:
    -----------
    human_df : pd.DataFrame
        Human ratings data
    llm_data_by_prompt : dict
        LLM data nested by prompt and model
    prompt : str, default "example"
        Which prompt to use for the comparison
    models : list, optional
        List of model names. If None, uses ["claude", "gemini", "gpt"]
    """

    if models is None:
        models = ["claude", "gemini", "gpt"]

    fig, axes = plt.subplots(
        nrows=1, ncols=len(models),
        figsize=(5 * len(models), 4),
        sharey=True
    )

    # Handle case where there's only one model
    if len(models) == 1:
        axes = [axes]

    attributes = sorted(human_df["attribute"].unique())
    colors = dict(zip(attributes, plt.cm.tab10.colors))

    for col, model in enumerate(models):
        ax = axes[col]
        llm_df = llm_data_by_prompt[prompt][model]

        df = prepare_hlc_dataframe(human_df, llm_df)
        df["residual"] = df["rating_llm"] - df["rating_human"]

        for attr in attributes:
            sub = df[df["attribute"] == attr]
            ax.scatter(
                sub["rating_human"],
                sub["residual"],
                color=colors[attr],
                alpha=0.75,
                label=attr if col == 0 else None
            )

        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_title(model)
        ax.set_xlabel("Human mean rating")

        if col == 0:
            ax.set_ylabel("LLM − Human")

    fig.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=colors[a], label=a, markersize=8)
            for a in attributes
        ],
        loc="lower center",
        ncol=min(len(attributes), len(models)),
        frameon=False
    )

    fig.suptitle(
        f"HLC Residuals (LLM − Human)\nPrompt = {prompt}",
        fontsize=14
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.9])
    plt.show()

# --------------------------------------------------
# Plot A: ESR vs Human Effect Strength
# --------------------------------------------------

def plot_esr_vs_human_strength(esr_df, esr_col, title, prompts=None, models=None):
    """
    Plot ESR vs human effect strength.
    
    Parameters:
    -----------
    esr_df : pd.DataFrame
        Must contain: attribute, model, prompt, human_effect, <esr_col>
    esr_col : str
        Name of the ESR column to plot
    title : str
        Plot title
    prompts : list, optional
        List of prompt names. If None, uses ["minimal", "example"]
    models : list, optional
        List of model names. If None, uses sorted unique values from esr_df
    """

    import matplotlib.pyplot as plt

    if prompts is None:
        prompts = ["minimal", "example"]
    if models is None:
        models = sorted(esr_df["model"].unique())

    plt.figure(figsize=(7, 5))

    # Create markers for prompts (cycle through available markers if needed)
    available_markers = ["o", "^", "s", "D", "v", "<", ">", "p", "*", "h"]
    markers = dict(zip(prompts, available_markers[:len(prompts)]))
    
    colors = dict(zip(models, plt.cm.Set1.colors))

    for model in models:
        for prompt in prompts:
            sub = esr_df[
                (esr_df["model"] == model) &
                (esr_df["prompt"] == prompt)
            ]

            plt.scatter(
                sub["human_effect"],
                sub[esr_col],
                label=f"{model} – {prompt}",
                marker=markers[prompt],
                color=colors[model],
                alpha=0.75
            )

    plt.axhline(1, linestyle="--", color="black")
    plt.xlabel("Human effect strength")
    plt.ylabel(esr_col)
    plt.title(title)

    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

# --------------------------------------------------
# Plot B: ESR heatmap (centered at ESR = 1)
# --------------------------------------------------

# --------------------------------------------------
# Plot ESR heatmap (centered at ESR = 1, with options)
# --------------------------------------------------

def plot_esr_heatmap(
    esr_df,
    esr_col,
    title,
    drop_non_significant=False,
    significant_main_attrs=None,
    significant_interaction_attrs=None,
    model_order=None,
    prompt_order=None
):
    """
    ESR heatmap with:
      - white at ESR = 1 (optimal)
      - blue below 1 (attenuation)
      - red above 1 (exaggeration)
      - values clipped to [0, 5]

    Expects an ``effect_type`` column ('main' or 'interaction') in *esr_df*.
    Each (attribute, effect_type) pair becomes its own row, so 5 main +
    5 interaction attributes yield 10 rows.

    Parameters
    ----------
    drop_non_significant : bool
        If True, keep only rows whose attribute belongs to the
        corresponding significant set for its effect type.
    significant_main_attrs : set or list, optional
        Attributes with significant human main effects.
    significant_interaction_attrs : set or list, optional
        Attributes with significant human interaction effects.
    model_order : list or tuple, optional
        Order of models. If None, uses sorted unique values from esr_df.
    prompt_order : list or tuple, optional
        Order of prompts. If None, uses sorted unique values from esr_df.
    """

    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

    if model_order is None:
        model_order = sorted(esr_df["model"].unique())
    if prompt_order is None:
        prompt_order = sorted(esr_df["prompt"].unique())

    df = esr_df.copy()

    # -----------------------------
    # Effect-type-aware filtering
    # -----------------------------
    if drop_non_significant:
        if significant_main_attrs is None or significant_interaction_attrs is None:
            raise ValueError(
                "significant_main_attrs and significant_interaction_attrs "
                "must both be provided when drop_non_significant=True"
            )
        sig_main = set(significant_main_attrs)
        sig_inter = set(significant_interaction_attrs)
        mask = (
            ((df["effect_type"] == "main") & df["attribute"].isin(sig_main))
            | ((df["effect_type"] == "interaction") & df["attribute"].isin(sig_inter))
        )
        df = df[mask]

    # -----------------------------
    # Row label: abbreviated attribute + short effect type
    # -----------------------------
    attr_abbrev = {
        "knowledgeable": "knowl.",
        "well-prepared": "well-prep.",
    }
    effect_abbrev = {"main": "main", "interaction": "int."}
    df["row_label"] = (
        df["attribute"].map(lambda a: attr_abbrev.get(a, a))
        + " ("
        + df["effect_type"].map(lambda e: effect_abbrev.get(e, e))
        + ")"
    )

    # -----------------------------
    # Pivot table
    # -----------------------------
    pivot = df.pivot_table(
        index="row_label",
        columns=["model", "prompt"],
        values=esr_col
    )

    # -----------------------------
    # Enforce row order: main effects first, then interactions
    # -----------------------------
    main_labels = sorted(a for a in pivot.index if a.endswith("(main)"))
    inter_labels = sorted(a for a in pivot.index if a.endswith("(int.)"))
    ordered_rows = main_labels + inter_labels
    pivot = pivot.loc[ordered_rows]

    # -----------------------------
    # Enforce column order: model → prompt
    # -----------------------------
    ordered_cols = [
        (m, p)
        for m in model_order
        for p in prompt_order
        if (m, p) in pivot.columns
    ]
    pivot = pivot[ordered_cols]

    pivot_clipped = pivot.clip(lower=0, upper=5)

    # -----------------------------
    # Colormap: blue → white → red
    # -----------------------------
    cmap = LinearSegmentedColormap.from_list(
        "esr_diverging",
        ["#2166ac", "white", "#b2182b"]
    )

    norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=5)

    # -----------------------------
    # Plot
    # -----------------------------
    n_rows = len(pivot_clipped)
    fig_height = max(4, 0.45 * n_rows + 1.5)
    plt.figure(figsize=(8, fig_height))
    im = plt.imshow(
        pivot_clipped.values,
        cmap=cmap,
        norm=norm,
        aspect="auto"
    )

    cbar = plt.colorbar(im)
    cbar.set_label("Effect Size Ratio (ESR)")
    cbar.set_ticks([0, 0.5, 1, 2, 3, 4, 5])
    cbar.set_ticklabels(["0", "0.5", "1", "2", "3", "4", "≥5"])

    # Annotate cells with numeric values
    data = pivot_clipped.values
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                continue
            text_color = "white" if val < 0.3 or val > 3.5 else "black"
            plt.text(j, i, f"{val:.2f}", ha="center", va="center",
                     color=text_color, fontsize=7)

    plt.yticks(
        range(len(pivot_clipped.index)),
        pivot_clipped.index
    )

    plt.xticks(
        range(len(pivot_clipped.columns)),
        [f"{m}\n{p}" for m, p in pivot_clipped.columns],
        rotation=45,
        ha="right"
    )

    plt.title(title)
    plt.tight_layout()
    plt.show()




