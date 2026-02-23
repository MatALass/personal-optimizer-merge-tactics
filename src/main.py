from dataclasses import dataclass
from typing import List, Dict, Tuple, Iterable, Optional
import pandas as pd
import numpy as np

# ============ 1) Modèle de bonus par trait (breadth-only: palier 2 uniquement) ============
@dataclass(frozen=True)
class TraitScheme:
    thresholds: Dict[int, float]          # ex {2: 20}
    breadth_only: bool = True             # ne récompense que le palier minimal (≥2)

    def bonus_for_count(self, n: int) -> float:
        if n < 2:
            return 0.0
        if self.breadth_only:
            base_threshold = min(self.thresholds.keys())
            return self.thresholds[base_threshold]
        best = [b for t, b in self.thresholds.items() if n >= t]
        return max(best) if best else 0.0


# ============ 2) Normalisation des cartes ============
def normalize_cards_df(cards: pd.DataFrame, traits_col: str = "traits") -> pd.DataFrame:
    df = cards.copy()

    def to_list(x):
        if isinstance(x, list):
            return [t.strip() for t in x]
        if isinstance(x, str):
            return [t.strip() for t in x.split(";") if t.strip()]
        return []

    if traits_col not in df.columns:
        df[traits_col] = [[]]
    df[traits_col] = df[traits_col].apply(to_list)

    if "base_power" not in df.columns:
        df["base_power"] = 1.0
    if "elixir" not in df.columns:
        df["elixir"] = 0

    df["card"] = df["card"].astype(str)
    df["base_power"] = pd.to_numeric(df["base_power"], errors="coerce").fillna(0.0)
    df["elixir"] = pd.to_numeric(df["elixir"], errors="coerce").fillna(0)
    return df


# ============ 3) Score: priorité au nombre de traits activés (≥2) ============
def composition_score(cards_df: pd.DataFrame,
                      idxs: Iterable[int],
                      trait_schemes: Dict[str, TraitScheme],
                      initial_trait_counts: Dict[str, int],
                      weight_pairs: float = 1e6) -> Tuple[float, Dict]:
    """
    Score = priorité absolue au nombre de traits activés (≥2).
    Tie-break: élixir total (plus faible = mieux).
    """
    idxs = list(idxs)
    subset = cards_df.iloc[idxs]

    # Compter les occurrences des traits de la compo
    counts = {trait: int(subset["traits"].apply(lambda ts: trait in ts).sum())
              for trait in trait_schemes.keys()}

    # Bonus (ici seulement palier 2 si breadth_only=True)
    per_trait_bonus = {}
    for trait, scheme in trait_schemes.items():
        c = counts[trait] + int(initial_trait_counts.get(trait, 0))
        per_trait_bonus[trait] = scheme.bonus_for_count(c)

    # Nombre de traits activés (≥2)
    pairs2 = sum(1 for v in per_trait_bonus.values() if v > 0)

    # Tie-break: élixir total
    total_elixir = float(subset["elixir"].sum())
    score = weight_pairs * pairs2 - total_elixir * 1e-3

    details = {
        "team_cards": list(subset["card"].values),
        "team_size": len(subset),
        "total_elixir": total_elixir,
        "base_power_sum": float(subset["base_power"].sum()),
        "trait_counts": counts,
        "initial_trait_counts": initial_trait_counts,
        "per_trait_bonus": per_trait_bonus,
        "pairs2": pairs2,
    }
    return score, details


# ============ 4) Beam search orienté "paires" + locked/banned ============
def top_compositions(cards: pd.DataFrame,
                     team_size: int = 6,
                     max_elixir: Optional[float] = None,
                     top_k: int = 50,
                     beam_width: int = 2000,
                     trait_schemes: Optional[Dict[str, TraitScheme]] = None,
                     initial_trait_counts: Optional[Dict[str, int]] = None,
                     prefer_traits_for_heuristic: Optional[List[str]] = None,
                     target_pairs2_min: int = 6,
                     locked_cards: Optional[List[str]] = None,
                     banned_cards: Optional[List[str]] = None) -> pd.DataFrame:

    cards_df = normalize_cards_df(cards)

    # Interdits
    if banned_cards:
        banned_set = set(map(str.lower, banned_cards))
        cards_df = cards_df[~cards_df["card"].str.lower().isin(banned_set)].reset_index(drop=True)

    n = len(cards_df)
    if n < team_size:
        raise ValueError(f"Pas assez de cartes ({n}) pour une team de taille {team_size}.")

    if trait_schemes is None:
        trait_schemes = {"Avenger": TraitScheme({2: 20}, breadth_only=True)}
    if initial_trait_counts is None:
        initial_trait_counts = {}
    if prefer_traits_for_heuristic is None:
        prefer_traits_for_heuristic = list(trait_schemes.keys())
    if locked_cards is None:
        locked_cards = []

    # Locked mapping
    name_to_idx = {name.lower(): i for i, name in enumerate(cards_df["card"].tolist())}
    locked_idxs: List[int] = []
    for nm in locked_cards:
        key = nm.lower()
        if key not in name_to_idx:
            raise ValueError(f"Carte obligatoire introuvable: '{nm}'")
        locked_idxs.append(name_to_idx[key])
    locked_idxs = sorted(set(locked_idxs))
    if len(locked_idxs) > team_size:
        raise ValueError(f"{len(locked_idxs)} cartes obligatoires > team_size={team_size}.")

    # Heuristique : favoriser la fermeture des paires (0->1 un peu, 1->2 beaucoup)
    init_counts = {t: int(initial_trait_counts.get(t, 0)) for t in trait_schemes.keys()}

    def card_help_score(ts: List[str], counts_like: Dict[str, int]) -> int:
        s = 0
        for t in ts:
            if t in trait_schemes:
                if counts_like.get(t, 0) == 1:
                    s += 3  # fermer une paire
                elif counts_like.get(t, 0) == 0:
                    s += 1  # ouvrir une paire potentielle
        return s

    help_vals = cards_df["traits"].apply(lambda ts: card_help_score(ts, init_counts)).values
    heuristic = help_vals.astype(float)  # base_power ignoré
    order = np.argsort(-heuristic)

    # Borne sup: nb max de traits (≥2) atteignables
    def upper_bound(partial_idxs: List[int], next_pos: int) -> float:
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
        return optimistic_pairs

    # Beam init : locked déjà inclus (ordonnés selon 'order' pour respecter l'invariance)
    if locked_idxs:
        pos_map = {idx: int(np.where(order == idx)[0][0]) for idx in locked_idxs}
        pre_idxs = [idx for idx, _ in sorted(pos_map.items(), key=lambda x: x[1])]
        last_pos = max(pos_map.values())
        start_after = last_pos + 1
        beam = [(pre_idxs, upper_bound(pre_idxs, start_after))]
        start_pos_init = start_after
    else:
        beam = [([], upper_bound([], 0))]
        start_pos_init = 0

    # Beam search
    for _ in range(team_size - len(locked_idxs)):
        new_beam = []
        for partial_idxs, _ub in beam:
            if not partial_idxs:
                start_pos = start_pos_init
            else:
                pos_of_last = int(np.where(order == partial_idxs[-1])[0][0])
                start_pos = pos_of_last + 1
            for pos in range(start_pos, len(order)):
                idx = order[pos]
                if idx in partial_idxs or idx in locked_idxs:
                    continue
                candidate = partial_idxs + [idx]
                if max_elixir is not None:
                    cost = float(cards_df.iloc[candidate]["elixir"].sum())
                    if cost > max_elixir:
                        continue
                ub = upper_bound(candidate, pos + 1)
                new_beam.append((candidate, ub))
        if not new_beam:
            break
        new_beam.sort(key=lambda x: -x[1])
        beam = new_beam[:beam_width]

    # Évaluation finale
    scored: List[Tuple[float, Dict, List[int]]] = []
    for idxs, _ in beam:
        if any(li not in idxs for li in locked_idxs):
            continue
        if len(idxs) != team_size:
            continue
        score, details = composition_score(
            cards_df, idxs,
            trait_schemes=trait_schemes,
            initial_trait_counts=initial_trait_counts
        )
        scored.append((score, details, idxs))

    if not scored:
        # fallback : locked + meilleurs heuristiques
        base_idxs = locked_idxs[:]
        for i in order:
            if i not in base_idxs:
                base_idxs.append(i)
            if len(base_idxs) == team_size:
                break
        score, details = composition_score(
            cards_df, base_idxs,
            trait_schemes=trait_schemes,
            initial_trait_counts=initial_trait_counts
        )
        scored = [(score, details, base_idxs)]

    # Filtre “au moins X traits activés”
    scored_filtered = [x for x in scored if x[1]["pairs2"] >= target_pairs2_min]
    if scored_filtered:
        scored = scored_filtered

    # Tri: 1) pairs2 desc, 2) score (tie-break élixir) desc
    scored.sort(key=lambda x: (x[1]["pairs2"], x[0]), reverse=True)

    # Résultat
    top = scored[:top_k]
    rows = []
    for rank, (score, details, idxs) in enumerate(top, start=1):
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


# ============ 5) Jeu de traits (tu peux en ajouter/en enlever) ============
trait_schemes = {
    "Avenger":   TraitScheme({2: 20}, breadth_only=True),
    "Goblin":    TraitScheme({2: 20}, breadth_only=True),
    "Brawler":   TraitScheme({2: 20}, breadth_only=True),
    "Juggernaut":TraitScheme({2: 20}, breadth_only=True),
    "Ranger":    TraitScheme({2: 20}, breadth_only=True),
    "Blaster":   TraitScheme({2: 20}, breadth_only=True),
    "Assassin":  TraitScheme({2: 20}, breadth_only=True),
    "Clan":      TraitScheme({2: 20}, breadth_only=True),
    "Noble":     TraitScheme({2: 20}, breadth_only=True),
    "Undead":    TraitScheme({2: 20}, breadth_only=True),
    "Ace":       TraitScheme({2: 20}, breadth_only=True),
    "Mage":      TraitScheme({2: 20}, breadth_only=True),
    "Fire":      TraitScheme({2: 20}, breadth_only=True),
    "Electric":  TraitScheme({2: 20}, breadth_only=True),
}

# ============ 6) Dataset cartes (mets les tiennes au besoin) ============
data = [
    {"card": "Knight",            "elixir": 2, "base_power": 6,  "traits": "Noble;Juggernaut"},
    {"card": "Archers",           "elixir": 2, "base_power": 6,  "traits": "Clan;Ranger"},
    {"card": "Goblins",           "elixir": 2, "base_power": 6,  "traits": "Goblin;Assassin"},
    {"card": "Spear Goblins",     "elixir": 2, "base_power": 6,  "traits": "Goblin;Blaster"},
    {"card": "Barbarians",        "elixir": 2, "base_power": 6,  "traits": "Clan;Brawler"},
    {"card": "Skeleton Dragons",  "elixir": 2, "base_power": 6,  "traits": "Undead;Ranger"},
    {"card": "Wizard",            "elixir": 2, "base_power": 6,  "traits": "Fire;Mage"},

    {"card": "Musketeer",         "elixir": 3, "base_power": 8,  "traits": "Noble;Blaster"},
    {"card": "Valkyrie",          "elixir": 3, "base_power": 8,  "traits": "Clan;Juggernaut"},
    {"card": "P.E.K.K.A",         "elixir": 3, "base_power": 8,  "traits": "Ace;Avenger"},
    {"card": "Prince",            "elixir": 3, "base_power": 8,  "traits": "Noble;Brawler"},
    {"card": "Giant Skeleton",    "elixir": 3, "base_power": 8,  "traits": "Undead;Brawler"},
    {"card": "Dart Goblin",       "elixir": 3, "base_power": 8,  "traits": "Goblin;Ranger"},
    {"card": "Electro Giant",     "elixir": 3, "base_power": 8,  "traits": "Electric;Avenger"},
    {"card": "Executioner",       "elixir": 3, "base_power": 8,  "traits": "Ace;Blaster"},

    {"card": "Witch",             "elixir": 4, "base_power": 10, "traits": "Undead;Avenger"},
    {"card": "Baby Dragon",       "elixir": 4, "base_power": 10, "traits": "Fire;Blaster"},
    {"card": "Princess",          "elixir": 4, "base_power": 10, "traits": "Noble;Ranger"},
    {"card": "Electro Wizard",    "elixir": 4, "base_power": 10, "traits": "Electric;Mage"},
    {"card": "Mega Knight",       "elixir": 4, "base_power": 10, "traits": "Ace;Brawler"},
    {"card": "Royal Ghost",       "elixir": 4, "base_power": 10, "traits": "Undead;Assassin"},
    {"card": "Bandit",            "elixir": 4, "base_power": 10, "traits": "Ace;Avenger"},
    {"card": "Goblin Machine",    "elixir": 4, "base_power": 10, "traits": "Goblin;Juggernaut"},

    {"card": "Skeleton King",     "elixir": 5, "base_power": 12, "traits": "Undead;Juggernaut"},
    {"card": "Golden Knight",     "elixir": 5, "base_power": 12, "traits": "Noble;Assassin"},
    {"card": "Archer Queen",      "elixir": 5, "base_power": 12, "traits": "Clan;Avenger"},
]
cards = pd.DataFrame(data)


# ============ 7) Interaction: demander 2 traits initiaux + locked/banned (optionnels) ============
def ask_initial_traits(trait_schemes: Dict[str, TraitScheme]) -> Dict[str, int]:
    """Demande deux traits initiaux (Entrée = rien). Même trait deux fois => compte 2."""
    all_traits = list(trait_schemes.keys())
    traits_set = {t.lower(): t for t in all_traits}

    print("\n--- Choisis tes traits initiaux (2 entrées). Appuie Entrée pour 'rien'. ---")
    print("Traits disponibles :", ", ".join(all_traits))

    init_counts: Dict[str, int] = {}
    for i in range(1, 3):
        raw = input(f"Trait {i} (ou vide) : ").strip()
        if not raw:
            continue
        key = raw.lower()
        if key not in traits_set:
            print(f"  → '{raw}' inconnu. Ignoré. (Vérifie l'orthographe)")
            print("    Rappel traits :", ", ".join(all_traits))
            continue
        canonical = traits_set[key]
        init_counts[canonical] = init_counts.get(canonical, 0) + 1

    if init_counts:
        print("→ Traits initiaux pris en compte :", init_counts)
    else:
        print("→ Aucun trait initial.")
    return init_counts


def ask_locked_and_banned(cards_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Demande (optionnel) une carte obligatoire et des cartes interdites."""
    available = cards_df["card"].tolist()
    avail_lower = {c.lower(): c for c in available}

    print("\n--- Carte obligatoire (optionnel) ---")
    print("Cartes disponibles :", ", ".join(available))
    raw_locked = input("Carte obligatoire (vide si aucune) : ").strip()
    locked: List[str] = []
    if raw_locked:
        key = raw_locked.lower()
        if key in avail_lower:
            locked = [avail_lower[key]]
        else:
            print(f"  → '{raw_locked}' introuvable. Aucune carte obligatoire appliquée.")

    print("\n--- Cartes interdites (optionnel) ---")
    raw_banned = input("Cartes interdites séparées par des virgules (vide si aucune) : ").strip()
    banned: List[str] = []
    if raw_banned:
        for token in [t.strip() for t in raw_banned.split(",") if t.strip()]:
            key = token.lower()
            if key in avail_lower:
                banned.append(avail_lower[key])
            else:
                print(f"  → '{token}' introuvable. Ignoré.")
    return locked, banned


# ============ 8) Main ============
if __name__ == "__main__":
    # 1) Demander traits initiaux (Entrée = rien)
    initial_trait_counts = ask_initial_traits(trait_schemes)

    # 2) Objectif min de paires activées : 6 sans bonus init, 7 avec
    min_pairs = 6 if not initial_trait_counts else 7

    # 3) (Optionnel) carte obligatoire + interdits
    locked_cards, banned_cards = ask_locked_and_banned(cards)

    # 4) Recherche
    result = top_compositions(
        cards=cards,
        team_size=6,
        max_elixir=None,
        top_k=20,
        beam_width=2000,  # 2000–3000 si tu veux être tranquille
        trait_schemes=trait_schemes,
        initial_trait_counts=initial_trait_counts,
        prefer_traits_for_heuristic=list(trait_schemes.keys()),
        target_pairs2_min=min_pairs,
        locked_cards=locked_cards,
        banned_cards=banned_cards
    )

    best = result.iloc[0]
    print("\n=== Résultat ===")
    print("Pairs>=2:", best["pairs2"])
    print("Team:", best["team"])
    print("Traits activés:", [t for t, v in best["per_trait_bonus"].items() if v > 0])
    if locked_cards:
        print("Carte obligatoire:", locked_cards)
    if banned_cards:
        print("Cartes interdites:", banned_cards)
