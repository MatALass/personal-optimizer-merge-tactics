from __future__ import annotations
from typing import Dict, Iterable, Tuple
from collections import Counter
import pandas as pd

from .models import TraitScheme


def composition_score(
    cards_df: pd.DataFrame,
    idxs: Iterable[int],
    trait_schemes: Dict[str, TraitScheme],
    initial_trait_counts: Dict[str, int],
    weight_pairs: float = 1e6,
) -> Tuple[float, Dict]:
    """
    Primary objective: maximize number of activated traits (>=2).
    Tie-break: lower total elixir is better (very small weight compared to activated traits).
    """
    idxs = list(idxs)
    subset = cards_df.iloc[idxs]

    # Count traits efficiently
    all_traits = []
    for ts in subset["traits"].tolist():
        all_traits.extend(ts)
    counter = Counter(all_traits)

    counts = {trait: int(counter.get(trait, 0)) for trait in trait_schemes.keys()}

    per_trait_bonus: Dict[str, float] = {}
    for trait, scheme in trait_schemes.items():
        c = counts[trait] + int(initial_trait_counts.get(trait, 0))
        per_trait_bonus[trait] = scheme.bonus_for_count(c)

    pairs2 = sum(1 for v in per_trait_bonus.values() if v > 0)

    total_elixir = float(subset["elixir"].sum())
    score = weight_pairs * pairs2 - total_elixir * 1e-3

    details = {
        "team_cards": list(subset["card"].values),
        "team_size": len(subset),
        "total_elixir": total_elixir,
        "base_power_sum": float(subset["base_power"].sum()),
        "trait_counts": counts,
        "initial_trait_counts": dict(initial_trait_counts),
        "per_trait_bonus": per_trait_bonus,
        "pairs2": pairs2,
    }
    return score, details