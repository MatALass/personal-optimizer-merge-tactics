from __future__ import annotations
from typing import List
import pandas as pd


def _traits_to_list(x) -> List[str]:
    if isinstance(x, list):
        return [str(t).strip() for t in x if str(t).strip()]
    if isinstance(x, str):
        return [t.strip() for t in x.split(";") if t.strip()]
    return []


def normalize_cards_df(cards: pd.DataFrame, traits_col: str = "traits") -> pd.DataFrame:
    """
    Normalizes schema:
    - Ensures columns: card (str), traits (list[str]), base_power (float), elixir (int)
    - Accepts traits as list or semicolon-separated string.
    """
    df = cards.copy()

    if "card" not in df.columns:
        raise ValueError("Missing required column: 'card'")

    if traits_col not in df.columns:
        # IMPORTANT: one empty list per row (not a single list for the whole column)
        df[traits_col] = [[] for _ in range(len(df))]

    df[traits_col] = df[traits_col].apply(_traits_to_list)

    if "base_power" not in df.columns:
        df["base_power"] = 1.0
    if "elixir" not in df.columns:
        df["elixir"] = 0

    df["card"] = df["card"].astype(str)
    df["base_power"] = pd.to_numeric(df["base_power"], errors="coerce").fillna(0.0).astype(float)
    df["elixir"] = pd.to_numeric(df["elixir"], errors="coerce").fillna(0).astype(int)

    return df.reset_index(drop=True)


def load_cards_csv(path: str) -> pd.DataFrame:
    """
    CSV columns expected:
      card, elixir, base_power, traits (semicolon-separated)
    """
    df = pd.read_csv(path)
    return normalize_cards_df(df, traits_col="traits")