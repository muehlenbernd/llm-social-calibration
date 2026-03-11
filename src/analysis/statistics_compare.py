import pandas as pd
import numpy as np
import os
import re
import statsmodels.formula.api as smf
from correlation_analysis import compute_sem_contrasts


PHASE2_EFFECTS = {
    "competent": {
        "main": "utterance[T.precise]",
        "interaction": "context[T.high]:utterance[T.precise]",
    },
    "likeable": {
        "interaction": "context[T.high]:utterance[T.precise]",
    },
    "pedantic": {
        "main": "utterance[T.precise]",
    },
}


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def extract_coef(result, name):
    if name in result.params:
        return {
            "estimate": result.params[name],
            "se": result.bse[name],
            "z": result.tvalues[name],
            "p": result.pvalues[name],
        }
    else:
        return {
            "estimate": None,
            "se": None,
            "z": None,
            "p": None,
        }


def extract_effect_and_modulation(result, base_coef):
    """
    Extract a base effect and all prompt modulations of that effect.
    """
    rows = []

    # Base effect
    if base_coef in result.params:
        rows.append({
            "effect_level": "base",
            "prompt": "baseline",
            "estimate": result.params[base_coef],
            "se": result.bse[base_coef],
            "z": result.tvalues[base_coef],
            "p": result.pvalues[base_coef],
        })

    # Prompt modulations
    for name in result.params.index:
        if name.startswith(base_coef + ":prompt"):
            prompt = name.split("prompt[T.")[-1].replace("]", "")
            rows.append({
                "effect_level": "modulation",
                "prompt": prompt,
                "estimate": result.params[name],
                "se": result.bse[name],
                "z": result.tvalues[name],
                "p": result.pvalues[name],
            })

    return rows


def infer_prompt_label(path):
    """
    Infer prompt condition name from filename.
    Example:
        results_minimal_prompt.json -> 'minimal'
        results_sem_prompt.json     -> 'sem'
    """
    fname = os.path.basename(path)
    m = re.search(r"results_(.*?)_prompt", fname)
    if m is None:
        raise ValueError(f"Cannot infer prompt label from filename: {path}")
    return m.group(1)

def load_model_results(path):
    df = pd.read_json(path)
    df = df[df["valid"]]
    return df


def load_human_data(path):
    df = pd.read_excel(path)
    df = df.rename(columns={
        "Xscenario": "scenario",
        "Xcontext": "context",
        "Xanswer": "utterance",
        "Acomp": "competent",
        "Alike": "likeable",
        "Apednt": "pedantic",
    })
    df["context"] = (
        df["context"].str.lower().str.strip()
        .replace({"lowpr": "low", "highpr": "high"})
    )
    df["utterance"] = df["utterance"].str.lower().str.strip()
    return df


# ------------------------------------------------------------
# SEM constraint extraction
# ------------------------------------------------------------

def extract_constraints(contrasts,  is_human=False):
    """
    Extract the four SEM constraints as signed distances.
    """
    rows = []

    for _, r in contrasts.iterrows():
        delta_main = 0.5 * (r["delta_form_high"] + r["delta_form_low"])

        model_label = "human" if is_human else r["model"]

        if r["attribute"] == "competent":
            rows.append({
                "model": model_label,
                "scenario": r["scenario"],
                "constraint": "competent_main",
                "distance": delta_main
            })
            rows.append({
                "model": model_label,
                "scenario": r["scenario"],
                "constraint": "competent_interaction",
                "distance": r["delta_interaction"]
            })

        if r["attribute"] == "likeable":
            rows.append({
                "model": model_label,
                "scenario": r["scenario"],
                "constraint": "likeable_interaction",
                "distance": r["delta_interaction"]
            })

        if r["attribute"] == "pedantic":
            rows.append({
                "model": model_label,
                "scenario": r["scenario"],
                "constraint": "pedantic_main",
                "distance": delta_main
            })
        
        if r["attribute"] == "helpful":
            rows.append({
                "model": model_label,
                "scenario": r["scenario"],
                "constraint": "helpful_main",
                "distance": delta_main
            })
            rows.append({
                "model": model_label,
                "scenario": r["scenario"],
                "constraint": "helpful_interaction",
                "distance": r["delta_interaction"]
            })
        if r["attribute"] == "knowledgeable":
            rows.append({
                "model": model_label,
                "scenario": r["scenario"],
                "constraint": "knowledgeable_main",
                "distance": delta_main
            })
            rows.append({
                "model": model_label,
                "scenario": r["scenario"],
                "constraint": "knowledgeable_interaction",
                "distance": r["delta_interaction"]
            })
        if r["attribute"] == "well-prepared":
            rows.append({
                "model": model_label,
                "scenario": r["scenario"],
                "constraint": "well-prepared_main",
                "distance": delta_main
            })
            rows.append({
                "model": model_label,
                "scenario": r["scenario"],
                "constraint": "well-prepared_interaction",
                "distance": r["delta_interaction"]
            })

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Human signed distances (tertiary metric reference)
# ------------------------------------------------------------

def compute_human_signed_distances(human_df):
    rows = []

    for scenario in human_df["scenario"].unique():
        sub = human_df[human_df["scenario"] == scenario]

        def mean(attr, ctx, utt):
            return sub[(sub["context"] == ctx) & (sub["utterance"] == utt)][attr].mean()

        # Competent
        dm = 0.5 * (
            (mean("competent", "high", "precise") - mean("competent", "high", "approx")) +
            (mean("competent", "low", "precise")  - mean("competent", "low", "approx"))
        )
        di = (
            (mean("competent", "high", "precise") - mean("competent", "high", "approx")) -
            (mean("competent", "low", "precise")  - mean("competent", "low", "approx"))
        )

        rows.append({"scenario": scenario, "constraint": "competent_main", "human_dist": dm})
        rows.append({"scenario": scenario, "constraint": "competent_interaction", "human_dist": di})

        # Likeable
        li = (
            (mean("likeable", "high", "precise") - mean("likeable", "high", "approx")) -
            (mean("likeable", "low", "precise")  - mean("likeable", "low", "approx"))
        )
        rows.append({"scenario": scenario, "constraint": "likeable_interaction", "human_dist": li})

        # Pedantic
        pm = 0.5 * (
            (mean("pedantic", "high", "precise") - mean("pedantic", "high", "approx")) +
            (mean("pedantic", "low", "precise")  - mean("pedantic", "low", "approx"))
        )
        rows.append({"scenario": scenario, "constraint": "pedantic_main", "human_dist": pm})

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Evaluation layers
# ------------------------------------------------------------

def evaluate_primary(df):
    return (
        df.assign(satisfied=df["distance"] > 0)
          .groupby(["model", "constraint"])["satisfied"]
          .mean()
          .reset_index(name="primary_score")
    )


def evaluate_secondary(df):
    return (
        df.groupby(["model", "constraint"])["distance"]
          .mean()
          .reset_index(name="secondary_score")
    )


def evaluate_tertiary(df, human_df):
    merged = df.merge(
        human_df, on=["scenario", "constraint"], how="left"
    )
    merged["abs_error"] = np.abs(merged["distance"] - merged["human_dist"])

    return (
        merged.groupby(["model", "constraint"])["abs_error"]
        .mean()
        .reset_index(name="tertiary_error")
    )


# -----------------------------
# Metrics
# -----------------------------

def primary_score(df):
    return (
        df.assign(correct=df["distance"] > 0)
          .groupby(["model", "constraint"])["correct"]
          .mean()
          .reset_index(name="primary")
    )


def tertiary_score(df, human):
    merged = df.merge(human, on=["scenario", "constraint"])
    merged["abs_error"] = (merged["distance"] - merged["human_dist"]).abs()

    return (
        merged.groupby(["model", "constraint"])["abs_error"]
        .mean()
        .reset_index(name="tertiary")
    )

def delta_symbol(delta, better_if_positive=True, eps=1e-6):
    """
    Return '+', '0', or '-' depending on whether a change is
    an improvement, no change, or deterioration.

    better_if_positive:
        True  -> higher is better (primary)
        False -> lower is better (tertiary)
    """
    if abs(delta) < eps:
        return "0"

    if better_if_positive:
        return "+" if delta > 0 else "-"
    else:
        return "+" if delta < 0 else "-"


def scale_normalized_error(model_df, human_df):
    """
    Compute scale-normalized (affine-aligned) MAE between
    model and human signed distances.

    Returns:
        DataFrame with columns:
        [model, constraint, tertiary_B]
    """
    rows = []

    merged = model_df.merge(
        human_df, on=["scenario", "constraint"], how="left"
    )

    for (model, constraint), sub in merged.groupby(["model", "constraint"]):
        # Drop missing or degenerate cases
        sub = sub.dropna(subset=["distance", "human_dist"])
        if len(sub) < 2:
            continue

        x = sub["human_dist"].values
        y = sub["distance"].values

        # Fit affine map y ≈ a*x + b
        a, b = np.polyfit(x, y, deg=1)

        # Aligned prediction
        y_hat = a * x + b

        # Mean absolute residual
        err = np.mean(np.abs(y - y_hat))

        rows.append({
            "model": model,
            "constraint": constraint,
            "tnorm": err
        })

    return pd.DataFrame(rows)



# ------------------------------------------------------------
# Main comparison
# ------------------------------------------------------------

def compare_conditions(path_a, path_b, human_path):
    df_a = load_model_results(path_a)
    df_b = load_model_results(path_b)
    human = load_human_data(human_path)

    contrasts_a = compute_sem_contrasts(df_a)
    contrasts_b = compute_sem_contrasts(df_b)

    cons_a = extract_constraints(contrasts_a)
    cons_b = extract_constraints(contrasts_b)

    human_dists = compute_human_signed_distances(human)

    results = []

    for label, cons in [("A", cons_a), ("B", cons_b)]:
        primary = evaluate_primary(cons)
        secondary = evaluate_secondary(cons)
        tertiary = evaluate_tertiary(cons, human_dists)

        out = (
            primary.merge(secondary, on=["model", "attribute"])
                   .merge(tertiary, on=["model", "attribute"])
        )
        out["condition"] = label
        results.append(out)

    final = pd.concat(results)
    return final


def compare(resultA, resultB, resultHuman):
    # --- Extract constraints ---
    A = extract_constraints(compute_sem_contrasts(resultA))
    B = extract_constraints(compute_sem_contrasts(resultB))

    human = compute_human_signed_distances(resultHuman)

    # --- Primary metric ---
    P_A = primary_score(A).rename(columns={"primary": "primary_A"})
    P_B = primary_score(B).rename(columns={"primary": "primary_B"})

    # --- Absolute tertiary metric ---
    T_A = tertiary_score(A, human).rename(columns={"tertiary": "tertiary_A"})
    T_B = tertiary_score(B, human).rename(columns={"tertiary": "tertiary_B"})

    # --- Scale-normalized tertiary metric ---
    TN_A = scale_normalized_error(A, human).rename(columns={"tnorm": "tnorm_A"})
    TN_B = scale_normalized_error(B, human).rename(columns={"tnorm": "tnorm_B"})

    # --- Merge all metrics ---
    table = (
        P_A.merge(P_B, on=["model", "attribute"])
           .merge(T_A, on=["model", "attribute"])
           .merge(T_B, on=["model", "attribute"])
           .merge(TN_A, on=["model", "attribute"])
           .merge(TN_B, on=["model", "attribute"])
    )

    # --- Deltas ---
    table["Δ_primary"] = table["primary_B"] - table["primary_A"]
    table["Δ_tertiary"] = table["tertiary_B"] - table["tertiary_A"]
    table["Δ_tnorm"] = table["tnorm_B"] - table["tnorm_A"]

    # --- Directional symbols ---
    table["primary_change"] = table["Δ_primary"].apply(
        lambda d: delta_symbol(d, better_if_positive=True)
    )

    table["tertiary_change"] = table["Δ_tertiary"].apply(
        lambda d: delta_symbol(d, better_if_positive=False)
    )

    table["tnorm_change"] = table["Δ_tnorm"].apply(
        lambda d: delta_symbol(d, better_if_positive=False)
    )

    return table.sort_values(["attribute", "model"])

def run_phase2_mixed_effects_from_files(
    json_files,
    model_name,
    attribute_name,
):
    """
    Phase 2 analysis (LLM-only mixed-effects), using JSON result files
    to define prompt conditions automatically.

    Parameters
    ----------
    json_files : list[str]
        Paths to result JSON files (e.g. minimal, example, sem).
    model_name : str
        Model to analyze (e.g. 'gpt', 'claude', 'gemini').
    attribute_name : str
        Social attribute (e.g. 'competent').

    Returns
    -------
    fitted MixedLMResults object
    """

    dfs = []

    for path in json_files:
        prompt = infer_prompt_label(path)

        df = pd.read_json(path)
        df = df[df["valid"]].copy()

        df["prompt"] = prompt
        dfs.append(df)

    # Pool all prompt conditions
    data = pd.concat(dfs, ignore_index=True)

    # Subset model + attribute
    sub = data[
        (data["model"] == model_name)
        & (data["attribute"] == attribute_name)
    ].copy()

    # Categorical coding (important!)
    sub["context"] = pd.Categorical(
        sub["context"], categories=["low", "high"]
    )
    sub["utterance"] = pd.Categorical(
        sub["utterance"], categories=["approx", "precise"]
    )
    sub["prompt"] = pd.Categorical(
        sub["prompt"], categories=sorted(sub["prompt"].unique())
    )

    # Mixed-effects model
    formula = "likert ~ context * utterance * prompt"

    md = smf.mixedlm(
        formula,
        sub,
        groups=sub["scenario"],
        re_formula="1",
    )

    mdf = md.fit(method="lbfgs")

    return mdf

def compare_phase2_old(
    json_files,
    models,
    attributes,
):
    """
    Run Phase-2 mixed-effects analyses across all models and attributes.

    Parameters
    ----------
    json_files : list[str]
        Result files defining prompt conditions.
    models : list[str]
        LLM models to analyze.
    attributes : list[str]
        Social attributes to analyze.

    Returns
    -------
    DataFrame with Phase-2 interaction results.
    """

    rows = []

    for model in models:
        for attr in attributes:
            try:
                res = run_phase2_mixed_effects_from_files(
                    json_files=json_files,
                    model_name=model,
                    attribute_name=attr,
                )
            except Exception as e:
                print(f"Skipping {model} / {attr}: {e}")
                continue

            # --- Baseline SEM interaction ---
            base = extract_coef(
                res, "context[T.high]:utterance[T.precise]"
            )

            rows.append({
                "model": model,
                "attribute": attr,
                "effect": "SEM_interaction",
                "prompt": "baseline",
                **base,
            })

            # --- Prompt-modulated interactions ---
            for name in res.params.index:
                if "context[T.high]:utterance[T.precise]:prompt" in name:
                    prompt = name.split("prompt[T.")[-1].replace("]", "")

                    coef = extract_coef(res, name)

                    rows.append({
                        "model": model,
                        "attribute": attr,
                        "effect": "SEM_interaction_modulation",
                        "prompt": prompt,
                        **coef,
                    })

    return pd.DataFrame(rows)

def compare_phase2(
    json_files,
    models,
    attributes,
):
    """
    Phase-2 mixed-effects comparison aligned with the four
    theoretically relevant SEM effects.
    """

    rows = []

    for model in models:
        for attr in attributes:
            if attr not in PHASE2_EFFECTS:
                continue

            try:
                res = run_phase2_mixed_effects_from_files(
                    json_files=json_files,
                    model_name=model,
                    attribute_name=attr,
                )
            except Exception as e:
                print(f"Skipping {model} / {attr}: {e}")
                continue

            for effect_type, coef_name in PHASE2_EFFECTS[attr].items():
                extracted = extract_effect_and_modulation(res, coef_name)

                for r in extracted:
                    rows.append({
                        "model": model,
                        "attribute": attr,
                        "effect": f"{attr}_{effect_type}",
                        **r,
                    })

    return pd.DataFrame(rows)