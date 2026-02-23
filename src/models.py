from __future__ import annotations
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class TraitScheme:
    """
    Defines how a trait gives bonus depending on how many cards of that trait are present.

    thresholds: e.g. {2: 20, 4: 45}
    breadth_only=True means we only reward reaching the minimum threshold (>=2),
    which encourages activating many different traits (breadth).
    """
    thresholds: Dict[int, float]
    breadth_only: bool = True

    def bonus_for_count(self, n: int) -> float:
        if n < 2:
            return 0.0
        if self.breadth_only:
            base_threshold = min(self.thresholds.keys())
            return float(self.thresholds[base_threshold])
        best = [b for t, b in self.thresholds.items() if n >= t]
        return float(max(best)) if best else 0.0