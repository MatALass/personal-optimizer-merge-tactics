from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .io_data import normalize_cards_df
from .models import TraitScheme
from .scoring import composition_score


def top_compositions(
    cards: pd.DataFrame,
    team_size: int = 6,
    max_elixir: Optional[float] = None,
    top_k: int = 50,
    beam_width: int = 2000,
    trait_schemes: Optional[Dict[str, TraitScheme]] = None,
    initial_trait_counts: Optional[Dict[str, int]] = None,
    target_pairs2_min: int = 6,
    locked_cards: Optional[List[str]] = None,
    banned_cards: Optional[List[str]] = None,
) -> pd.DataFrame:
    cards_df = normalize_cards_df(cards)

    # banned filter
    if banned_cards:
        banned_set = set(map(str.lower, banned_cards))
        cards_df = cards_df[~cards_df["card"].str.lower().isin(banned_set)].reset_index(drop=True)

    n = len(cards_df)
    if n < team_size:
        raise ValueError(f"Not enough cards ({n}) for team_size={team_size}.")

    if trait_schemes is None:
        trait_schemes = {"Avenger": TraitScheme({2: 20}, breadth_only=True)}
    if initial_trait_counts is None:
        initial_trait_counts = {}
    if locked_cards is None:
        locked_cards = []

    # Locked mapping
    name_to_idx = {name.lower(): i for i, name in enumerate(cards_df["card"].tolist())}
    locked_idxs: List[int] = []
    for nm in locked_cards:
        key = nm.lower()
        if key not in name_to_idx:
            raise ValueError(f"Locked card not found: '{nm}'")
        locked_idxs.append(name_to_idx[key])
    locked_idxs = sorted(set(locked_idxs))
    if len(locked_idxs) > team_size:
        raise ValueError(f"{len(locked_idxs)} locked cards > team_size={team_size}.")

    # Heuristic: encourage closing pairs (1->2) strongly, opening pairs (0->1) mildly
    init_counts = {t: int(initial_trait_counts.get(t, 0)) for t in trait_schemes.keys()}

    def card_help_score(ts: List[str], counts_like: Dict[str, int]) -> int:
        s = 0
        for t in ts:
            if t in trait_schemes:
                if counts_like.get(t, 0) == 1:
                    s += 3
                elif counts_like.get(t, 0) == 0:
                    s += 1
        return s

    help_vals = cards_df["traits"].apply(lambda ts: card_help_score(ts, init_counts)).values
    order = np.argsort(-help_vals.astype(float))

    # Fast lookup: index -> position in order
    pos_in_order = {int(idx): int(pos) for pos, idx in enumerate(order)}

    def upper_bound(partial_idxs: List[int]) -> float:
        remaining = team_size - len(partial_idxs)
        if remaining <= 0:
            return 0.0
        subset = cards_df.iloc[partial_idxs] if partial_idxs else cards_df.iloc[[]]
        counts_now = {trait: int(subset["traits"].apply(lambda ts: trait in ts).sum())
                      for trait in trait_schemes.keys()}
        already_pairs = sum(
            1 for t, scheme in trait_schemes.items()
            if scheme.bonus_for_count(counts_now[t] + int(initial_trait_counts.get(t, 0))) > 0
        )
        optimistic_pairs = min(already_pairs + remaining * 2, len(trait_schemes))
        return float(optimistic_pairs)

    # Initialize beam with locked cards (kept in heuristic order)
    if locked_idxs:
        pre_idxs = sorted(locked_idxs, key=lambda idx: pos_in_order[idx])
        beam: List[Tuple[List[int], float]] = [(pre_idxs, upper_bound(pre_idxs))]
    else:
        beam = [([], upper_bound([]))]

    # Beam search expansion
    for _ in range(team_size - len(locked_idxs)):
        new_beam: List[Tuple[List[int], float]] = []
        for partial_idxs, _ub in beam:
            start_pos = 0
            if partial_idxs:
                start_pos = pos_in_order[partial_idxs[-1]] + 1

            for pos in range(start_pos, len(order)):
                idx = int(order[pos])
                if idx in partial_idxs or idx in locked_idxs:
                    continue

                candidate = partial_idxs + [idx]
                if max_elixir is not None:
                    cost = float(cards_df.iloc[candidate]["elixir"].sum())
                    if cost > max_elixir:
                        continue

                ub = upper_bound(candidate)
                new_beam.append((candidate, ub))

        if not new_beam:
            break

        new_beam.sort(key=lambda x: -x[1])
        beam = new_beam[:beam_width]

    # Final evaluation
    scored: List[Tuple[float, Dict, List[int]]] = []
    for idxs, _ in beam:
        if any(li not in idxs for li in locked_idxs):
            continue
        if len(idxs) != team_size:
            continue

        score, details = composition_score(
            cards_df,
            idxs,
            trait_schemes=trait_schemes,
            initial_trait_counts=initial_trait_counts,
        )
        scored.append((score, details, idxs))

    if not scored:
        # Fallback: locked + best heuristic fill
        base_idxs = locked_idxs[:]
        for i in order:
            i = int(i)
            if i not in base_idxs:
                base_idxs.append(i)
            if len(base_idxs) == team_size:
                break

        score, details = composition_score(
            cards_df,
            base_idxs,
            trait_schemes=trait_schemes,
            initial_trait_counts=initial_trait_counts,
        )
        scored = [(score, details, base_idxs)]

    # Filter by min activated traits
    scored_filtered = [x for x in scored if x[1]["pairs2"] >= target_pairs2_min]
    if scored_filtered:
        scored = scored_filtered

    # Sort by (pairs2 desc, score desc)
    scored.sort(key=lambda x: (x[1]["pairs2"], x[0]), reverse=True)

    rows = []
    for rank, (score, details, idxs) in enumerate(scored[:top_k], start=1):
        rows.append({
            "rank": rank,
            "score": score,
            "pairs2": details["pairs2"],
            "team": details["team_cards"],
            "team_size": details["team_size"],
            "total_elixir": details["total_elixir"],
            "trait_counts": details["trait_counts"],
            "initial_trait_counts": details["initial_trait_counts"],
            "per_trait_bonus": details["per_trait_bonus"],
            "locked_cards": locked_cards,
            "banned_cards": banned_cards or [],
        })

    return pd.DataFrame(rows)