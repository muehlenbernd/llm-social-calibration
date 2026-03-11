import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import os
import re


def infer_prompt_label(path):
    fname = os.path.basename(path)
    m = re.search(r"results_(.*?)_prompt", fname)
    if m is None:
        raise ValueError(f"Cannot infer prompt label from filename: {path}")
    return m.group(1)


def correlate_human_llm_grid(
    llm_json_files,
    human_xlsx_path,
    models,
    plot=False,
):
    """
    Compute human–LLM correlations across models × prompt strategies.

    Plot layout (if plot=True):
        rows    = models
        columns = prompt strategies

    Returns:
        DataFrame with columns:
        model, prompt, n_conditions, pearson_r, pearson_p,
        spearman_rho, spearman_p
    """

    # ---------------- Human data ----------------
    human = pd.read_excel(human_xlsx_path)

    


    
    human = human.rename(columns={
        "Xscenario": "scenario",
        "Xcontext": "context",
        "Xanswer": "utterance",
        "Acomp": "competent",
        "Alike": "likeable",
        "Apednt": "pedantic",
    })

    SCENARIO_MAP = {
    "bike": "bicycle",
    "cooking": "pasta",
    "absence": "office",
    # add others if needed
    }

    human["scenario"] = (
        human["scenario"]
        .str.lower()
        .str.strip()
        .replace(SCENARIO_MAP)
    )

    human["context"] = (
        human["context"]
        .str.lower()
        .str.strip()
        .replace({"lowpr": "low", "highpr": "high"})
    )
    human["utterance"] = human["utterance"].str.lower().str.strip()

    human_long = human.melt(
        id_vars=["scenario", "context", "utterance"],
        value_vars=["competent", "likeable", "pedantic"],
        var_name="attribute",
        value_name="human_rating",
    )

    human_means = (
        human_long
        .groupby(
            ["scenario", "context", "utterance", "attribute"],
            as_index=False
        )["human_rating"]
        .mean()
    .rename(columns={"human_rating": "human_mean"})
    )

    expected = 6 * 2 * 2 * 3  # 72
    if len(human_means) != expected:
        print(f"Warning: human cells = {len(human_means)}, expected {expected}")
    #human_means = (
    #    human_long
    #    .groupby(["scenario", "context", "utterance", "attribute"])
    #    ["human_rating"]
    #    .mean()
    #    .reset_index()
    #    .rename(columns={"human_rating": "human_mean"})
    #)

    # ---------------- Prepare storage ----------------
    prompt_labels = [infer_prompt_label(p) for p in llm_json_files]
    results = []
    plot_data = {}

    # ---------------- Loop ----------------
    for model in models:
        plot_data[model] = {}

        for path, prompt in zip(llm_json_files, prompt_labels):
            llm = pd.read_json(path)
            llm = llm[llm["valid"]]
            llm = llm[llm["model"] == model]

            if llm.empty:
                continue

            llm_means = (
                llm.groupby(
                    ["scenario", "context", "utterance", "attribute"]
                )["likert"]
                .mean()
                .reset_index()
                .rename(columns={"likert": "llm_mean"})
            )


            # 1. Check full key overlap
            human_keys = set(
                tuple(x) for x in
                human_means[["scenario", "context", "utterance", "attribute"]].values
            )

            llm_keys = set(
                tuple(x) for x in
                llm_means[["scenario", "context", "utterance", "attribute"]].values
            )

            merged = pd.merge(
                human_means,
                llm_means,
                on=["scenario", "context", "utterance", "attribute"],
                how="inner",
            )



            if len(merged) < 10:
                continue

            pearson_r, pearson_p = pearsonr(
                merged["human_mean"], merged["llm_mean"]
            )
            spearman_rho, spearman_p = spearmanr(
                merged["human_mean"], merged["llm_mean"]
            )

            n_conditions = (
                merged[["scenario", "context", "utterance", "attribute"]]
                .drop_duplicates()
                .shape[0]
            )
            
            results.append({
                "model": model,
                "prompt": prompt,
                "n_conditions": n_conditions,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_rho": spearman_rho,
                "spearman_p": spearman_p,
            })

            plot_data[model][prompt] = merged

    # ---------------- Plotting ----------------
    if plot:
        n_models = len(models)
        n_prompts = len(prompt_labels)

        # Set fixed window size (1200x900 pixels ≈ 12x9 inches at ~100 DPI)
        figsize = (12, 9)

        fig, axes = plt.subplots(
            n_models,
            n_prompts,
            figsize=figsize,
            sharex=True,
            sharey=True,
        )

        if n_models == 1:
            axes = np.expand_dims(axes, axis=0)
        if n_prompts == 1:
            axes = np.expand_dims(axes, axis=1)

        # Define colors for different attributes
        attribute_colors = {
            'competent': 'blue',
            'likeable': 'red', 
            'pedantic': 'green'
        }

        for i, model in enumerate(models):
            for j, prompt in enumerate(prompt_labels):
                ax = axes[i, j]

                if prompt not in plot_data.get(model, {}):
                    ax.axis("off")
                    continue

                df = plot_data[model][prompt]

                # Plot points with different colors for each attribute
                for attribute in df['attribute'].unique():
                    attr_df = df[df['attribute'] == attribute]
                    ax.scatter(
                        attr_df["human_mean"],
                        attr_df["llm_mean"],
                        alpha=0.65,
                        color=attribute_colors.get(attribute, 'gray'),
                        label=attribute if i == 0 and j == 0 else ""
                    )

                lo = min(df["human_mean"].min(), df["llm_mean"].min())
                hi = max(df["human_mean"].max(), df["llm_mean"].max())
                ax.plot([lo, hi], [lo, hi], linestyle="--", color="black")

                if i == 0:
                    ax.set_title(prompt)
                if j == 0:
                    ax.set_ylabel(f"{model}\nLLM mean")

                if i == n_models - 1:
                    ax.set_xlabel("Human mean")

                # Add legend only to the first subplot
                if i == 0 and j == 0:
                    ax.legend(title="Attribute", loc='upper left')

        plt.suptitle(
            "Human–LLM global alignment\n(rows = models, columns = prompt strategies)",
            y=1.02,
        )
        plt.tight_layout()
        plt.show()

    return pd.DataFrame(results)

def correlate_human_llm_models_prompts(
    llm_json_files,
    human_xlsx_path,
    models,
    plot=False,
):
    """
    Compute human–LLM correlations across multiple models and
    prompt strategies.

    Returns a DataFrame with rows:
      model × prompt
    """

    # ---------- Load and prepare human data ----------
    human = pd.read_excel(human_xlsx_path)

    human = human.rename(columns={
        "Xscenario": "scenario",
        "Xcontext": "context",
        "Xanswer": "utterance",
        "Acomp": "competent",
        "Alike": "likeable",
        "Apednt": "pedantic",
    })

    human["context"] = (
        human["context"]
        .str.lower()
        .str.strip()
        .replace({"lowpr": "low", "highpr": "high"})
    )
    human["utterance"] = human["utterance"].str.lower().str.strip()

    human_long = human.melt(
        id_vars=["scenario", "context", "utterance"],
        value_vars=["competent", "likeable", "pedantic"],
        var_name="attribute",
        value_name="human_rating",
    )

    human_means = (
        human_long
        .groupby(
            ["scenario", "context", "utterance", "attribute"]
        )["human_rating"]
        .mean()
        .reset_index()
        .rename(columns={"human_rating": "human_mean"})
    )

    results = []
    plot_store = {}

    # ---------- Loop over models ----------
    for model_name in models:
        plot_store[model_name] = []

        # ---------- Loop over prompts ----------
        for path in llm_json_files:
            prompt = infer_prompt_label(path)

            llm = pd.read_json(path)
            llm = llm[llm["valid"]]
            llm = llm[llm["model"] == model_name]

            if llm.empty:
                continue

            llm_means = (
                llm.groupby(
                    ["scenario", "context", "utterance", "attribute"]
                )["likert"]
                .mean()
                .reset_index()
                .rename(columns={"likert": "llm_mean"})
            )

            merged = pd.merge(
                human_means,
                llm_means,
                on=["scenario", "context", "utterance", "attribute"],
                how="inner",
            )

            if len(merged) < 10:
                continue

            pearson_r, pearson_p = pearsonr(
                merged["human_mean"], merged["llm_mean"]
            )
            spearman_rho, spearman_p = spearmanr(
                merged["human_mean"], merged["llm_mean"]
            )

            n_conditions = (
                merged[["scenario", "context", "utterance", "attribute"]]
                .drop_duplicates()
                .shape[0]
            )
   
         
            
            results.append({
                "model": model_name,
                "prompt": prompt,
                "n_conditions": n_conditions,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_rho": spearman_rho,
                "spearman_p": spearman_p,
            })

            plot_store[model_name].append((prompt, merged))

    # ---------- Plotting ----------
    if plot:
        n_models = len(plot_store)
        fig, axes = plt.subplots(
            1, n_models, figsize=(6 * n_models, 6), sharey=True
        )

        if n_models == 1:
            axes = [axes]

        for ax, (model_name, entries) in zip(axes, plot_store.items()):
            for prompt, df in entries:
                ax.scatter(
                    df["human_mean"],
                    df["llm_mean"],
                    alpha=0.6,
                    label=prompt,
                )

            lo = min(
                df["human_mean"].min() for _, df in entries
            )
            hi = max(
                df["human_mean"].max() for _, df in entries
            )
            ax.plot([lo, hi], [lo, hi], linestyle="--", color="black")

            ax.set_title(model_name)
            ax.set_xlabel("Human mean rating")
            ax.set_ylabel("LLM mean rating")
            ax.legend(title="Prompt")

        plt.suptitle("Human–LLM global alignment across models and prompts", y=1.02)
        plt.tight_layout()
        plt.show()

    return pd.DataFrame(results)



def correlate_human_llm(
    llm_json_path,
    human_xlsx_path,
    model_name,
    plot=False,
):
    """
    Compute global correlation between human and LLM mean ratings
    across all context × utterance × scenario × attribute conditions.

    Returns a dict with Pearson r and Spearman rho.
    """

    # ---------- Load LLM data ----------
    llm = pd.read_json(llm_json_path)
    llm = llm[llm["valid"]]

    llm = llm[llm["model"] == model_name]

    llm_means = (
        llm.groupby(
            ["scenario", "context", "utterance", "attribute"]
        )["likert"]
        .mean()
        .reset_index()
        .rename(columns={"likert": "llm_mean"})
    )

    # ---------- Load human data ----------
    human = pd.read_excel(human_xlsx_path)

    human = human.rename(columns={
        "Xscenario": "scenario",
        "Xcontext": "context",
        "Xanswer": "utterance",
        "Acomp": "competent",
        "Alike": "likeable",
        "Apednt": "pedantic",
    })

    human["context"] = (
        human["context"]
        .str.lower()
        .str.strip()
        .replace({"lowpr": "low", "highpr": "high"})
    )
    human["utterance"] = human["utterance"].str.lower().str.strip()

    human_long = human.melt(
        id_vars=["scenario", "context", "utterance"],
        value_vars=["competent", "likeable", "pedantic"],
        var_name="attribute",
        value_name="human_rating",
    )

    human_means = (
        human_long
        .groupby(
            ["scenario", "context", "utterance", "attribute"]
        )["human_rating"]
        .mean()
        .reset_index()
        .rename(columns={"human_rating": "human_mean"})
    )

    # ---------- Merge ----------
    merged = pd.merge(
        human_means,
        llm_means,
        on=["scenario", "context", "utterance", "attribute"],
        how="inner",
    )

    if len(merged) < 10:
        raise ValueError("Too few overlapping conditions to compute correlation.")

    # ---------- Correlations ----------
    pearson_r, pearson_p = pearsonr(
        merged["human_mean"], merged["llm_mean"]
    )
    spearman_rho, spearman_p = spearmanr(
        merged["human_mean"], merged["llm_mean"]
    )

    # ---------- Optional plot ----------
    if plot:
        plt.figure(figsize=(6, 6))
        plt.scatter(
            merged["human_mean"],
            merged["llm_mean"],
            alpha=0.7,
        )

        # identity line
        lo = min(merged["human_mean"].min(), merged["llm_mean"].min())
        hi = max(merged["human_mean"].max(), merged["llm_mean"].max())
        plt.plot([lo, hi], [lo, hi], linestyle="--")

        plt.xlabel("Human mean rating")
        plt.ylabel("LLM mean rating")
        plt.title(
            f"{model_name}: Human–LLM correlation\n"
            f"Pearson r = {pearson_r:.2f}, Spearman ρ = {spearman_rho:.2f}"
        )
        plt.tight_layout()
        plt.show()

    return {
        "model": model_name,
        "n_conditions": len(merged),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
    }


def compute_sem_contrasts(df: pd.DataFrame) -> pd.DataFrame:
    means = (
        df.groupby(
            ["model", "scenario", "context", "utterance", "attribute"],
            as_index=False
        )["likert"]
        .mean()
    )

    rows = []

    for (model, scenario, attribute), g in means.groupby(
        ["model", "scenario", "attribute"]
    ):
        try:
            p_h = g.query("context=='high' and utterance=='precise'")["likert"].iloc[0]
            a_h = g.query("context=='high' and utterance=='approx'")["likert"].iloc[0]
            p_l = g.query("context=='low' and utterance=='precise'")["likert"].iloc[0]
            a_l = g.query("context=='low' and utterance=='approx'")["likert"].iloc[0]
        except IndexError:
            continue

        rows.append({
            "model": model,
            "scenario": scenario,
            "attribute": attribute,
            "delta_form_high": p_h - a_h,
            "delta_form_low": p_l - a_l,
            "delta_interaction": (p_h - a_h) - (p_l - a_l)
        })

    return pd.DataFrame(rows)


def check_sem_constraints(df: pd.DataFrame) -> pd.DataFrame:
    def satisfies(row):
        if row["attribute"] in ["competent", "knowledgeable", "well-prepared"]:
            return row["delta_interaction"] > 0
        if row["attribute"] == "likeable":
            return row["delta_interaction"] < 0
        if row["attribute"] == "pedantic":
            return row["delta_form_high"] > 0 and row["delta_form_low"] > 0
        return None

    if df.empty:
        return df
    df["constraint_satisfied"] = df.apply(satisfies, axis=1)
    return df


def compute_form_effects(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute precise–approx form effects by context, scenario, attribute.
    """
    means = (
        df.groupby(
            ["model", "scenario", "context", "utterance", "attribute"],
            as_index=False
        )["likert"]
        .mean()
    )

    pivot = means.pivot_table(
        index=["model", "scenario", "attribute", "context"],
        columns="utterance",
        values="likert"
    ).reset_index()

    # assumes labels already normalized: precise / approx
    pivot["delta_form"] = pivot["precise"] - pivot["approx"]

    return pivot


def compute_interaction_effects(df_effects: pd.DataFrame) -> pd.DataFrame:
    """
    Compute interaction contrasts per scenario and attribute.
    """
    pivot = df_effects.pivot_table(
        index=["model", "scenario", "attribute"],
        columns="context",
        values="delta_form"
    ).reset_index()

    pivot["interaction"] = pivot["high"] - pivot["low"]
    return pivot


def bootstrap_interaction(
    df: pd.DataFrame,
    attribute: str,
    n_boot: int = 1000,
    random_state: int = 0
):
    """
    Bootstrap interaction effect over LLM samples.
    """
    rng = np.random.default_rng(random_state)
    df = df[df["attribute"] == attribute]

    boot_vals = []

    for _ in range(n_boot):
        boot_samples = []
        for _, group in df.groupby(["model", "scenario", "context", "utterance", "attribute"]):
            boot_samples.append(group.sample(len(group), replace=True, random_state=rng.integers(1e9)))
        boot = pd.concat(boot_samples)
        effects = compute_form_effects(boot)
        inter = compute_interaction_effects(effects)

        boot_vals.append(inter["interaction"].mean())

    return np.array(boot_vals)


def bootstrap_constraint_satisfaction(
    df: pd.DataFrame,
    attribute: str,
    n_boot: int = 1000,
    random_state: int = 0
):
    """
    Bootstrap probability that the SEM constraint is satisfied.
    """
    rng = np.random.default_rng(random_state)

    df = df[(df["valid"]) & (df["attribute"] == attribute)]

    results = []

    for _ in range(n_boot):
        boot_samples = []
        for _, group in df.groupby(["model", "scenario", "context", "utterance", "attribute"]):
            boot_samples.append(group.sample(len(group), replace=True, random_state=rng.integers(1e9)))
        boot = pd.concat(boot_samples)

        contrasts = compute_sem_contrasts(boot)
        contrasts = check_sem_constraints(contrasts)

        # Average constraint satisfaction across scenarios
        results.append(contrasts["constraint_satisfied"].mean())

    return np.array(results)

