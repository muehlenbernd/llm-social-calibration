import json
import pandas as pd


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_results(path: str) -> pd.DataFrame:
    return pd.read_json(path)


def export_for_mixed_effects(df: pd.DataFrame, path: str):
    df_valid = df[df["valid"]].copy()
    df_valid.to_csv(path, index=False)
