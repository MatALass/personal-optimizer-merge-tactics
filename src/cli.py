from __future__ import annotations
import argparse
from typing import Dict, List

from .models import TraitScheme
from .io_data import load_cards_csv
from .search import top_compositions


DEFAULT_TRAITS = {
    "Avenger":    TraitScheme({2: 20}, breadth_only=True),
    "Goblin":     TraitScheme({2: 20}, breadth_only=True),
    "Brawler":    TraitScheme({2: 20}, breadth_only=True),
    "Juggernaut": TraitScheme({2: 20}, breadth_only=True),
    "Ranger":     TraitScheme({2: 20}, breadth_only=True),
    "Blaster":    TraitScheme({2: 20}, breadth_only=True),
    "Assassin":   TraitScheme({2: 20}, breadth_only=True),
    "Clan":       TraitScheme({2: 20}, breadth_only=True),
    "Noble":      TraitScheme({2: 20}, breadth_only=True),
    "Undead":     TraitScheme({2: 20}, breadth_only=True),
    "Ace":        TraitScheme({2: 20}, breadth_only=True),
    "Mage":       TraitScheme({2: 20}, breadth_only=True),
    "Fire":       TraitScheme({2: 20}, breadth_only=True),
    "Electric":   TraitScheme({2: 20}, breadth_only=True),
}


def _parse_csv_list(s: str) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_initial_traits(s: str) -> Dict[str, int]:
    """
    Input format: "Goblin,Clan" (case-insensitive).
    If same trait appears twice -> count 2.
    """
    if not s:
        return {}
    tokens = _parse_csv_list(s)
    canonical = {t.lower(): t for t in DEFAULT_TRAITS.keys()}
    out: Dict[str, int] = {}
    for tok in tokens:
        key = tok.lower()
        if key not in canonical:
            raise ValueError(f"Unknown trait: '{tok}'. Available: {', '.join(DEFAULT_TRAITS.keys())}")
        t = canonical[key]
        out[t] = out.get(t, 0) + 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deck optimizer (synergy pairs) inspired by Clash Royale / auto-battler trait systems."
    )
    parser.add_argument("--cards", default="data/cards.csv", help="Path to cards CSV (default: data/cards.csv)")
    parser.add_argument("--team-size", type=int, default=6)
    parser.add_argument("--max-elixir", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--beam-width", type=int, default=2000)

    parser.add_argument("--initial-traits", default="", help='Example: "Goblin,Clan"')
    parser.add_argument("--locked", default="", help='Example: "Knight"')
    parser.add_argument("--banned", default="", help='Example: "Wizard,Witch"')

    args = parser.parse_args()

    cards_df = load_cards_csv(args.cards)

    initial_trait_counts = _parse_initial_traits(args.initial_traits)
    min_pairs = 6 if not initial_trait_counts else 7

    locked_cards = _parse_csv_list(args.locked)
    banned_cards = _parse_csv_list(args.banned)

    result = top_compositions(
        cards=cards_df,
        team_size=args.team_size,
        max_elixir=args.max_elixir,
        top_k=args.top_k,
        beam_width=args.beam_width,
        trait_schemes=DEFAULT_TRAITS,
        initial_trait_counts=initial_trait_counts,
        target_pairs2_min=min_pairs,
        locked_cards=locked_cards,
        banned_cards=banned_cards,
    )

    best = result.iloc[0]
    active_traits = [t for t, v in best["per_trait_bonus"].items() if v > 0]

    print("\n=== Best Team ===")
    print("pairs>=2:", int(best["pairs2"]))
    print("team:", best["team"])
    print("active traits:", active_traits)
    if locked_cards:
        print("locked:", locked_cards)
    if banned_cards:
        print("banned:", banned_cards)


if __name__ == "__main__":
    main()