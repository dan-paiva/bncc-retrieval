"""Reciprocal Rank Fusion (Cormack et al., 2009): junta rankings sem calibrar scores."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence


def rrf(rankings: Sequence[Sequence[str]], k: int = 60) -> list[str]:
    pontos: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for pos, codigo in enumerate(ranking, start=1):
            pontos[codigo] += 1 / (k + pos)
    return sorted(pontos, key=lambda c: (-pontos[c], c))
