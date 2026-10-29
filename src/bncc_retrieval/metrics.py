"""Métricas de ranking e intervalo de confiança por bootstrap.

Cada função recebe o ranking (códigos, do melhor para o pior) e o conjunto de códigos
corretos de uma consulta. A média sobre as consultas é feita fora.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def reciprocal_rank(ranking: Sequence[str], gold: set[str]) -> float:
    for pos, codigo in enumerate(ranking, start=1):
        if codigo in gold:
            return 1 / pos
    return 0.0


def hit_at(ranking: Sequence[str], gold: set[str], k: int) -> float:
    return float(any(c in gold for c in ranking[:k]))


def recall_at(ranking: Sequence[str], gold: set[str], k: int) -> float:
    return len(gold & set(ranking[:k])) / len(gold)


def ndcg_at(ranking: Sequence[str], gold: set[str], k: int) -> float:
    dcg = sum(1 / math.log2(pos + 1) for pos, c in enumerate(ranking[:k], start=1) if c in gold)
    ideal = sum(1 / math.log2(pos + 1) for pos in range(1, min(len(gold), k) + 1))
    return dcg / ideal


def por_consulta(ranking: Sequence[str], gold: Sequence[str]) -> dict[str, float]:
    g = set(gold)
    return {
        "mrr": reciprocal_rank(ranking, g),
        "hit@1": hit_at(ranking, g, 1),
        "hit@3": hit_at(ranking, g, 3),
        "hit@5": hit_at(ranking, g, 5),
        "hit@10": hit_at(ranking, g, 10),
        "ndcg@10": ndcg_at(ranking, g, 10),
    }


def bootstrap_ic(valores: Sequence[float], n: int = 10_000, alfa: float = 0.05, semente: int = 0):
    """Média e intervalo de confiança percentil (95% por padrão)."""
    v = np.asarray(valores, dtype=float)
    rng = np.random.default_rng(semente)
    medias = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.quantile(medias, alfa / 2)), float(np.quantile(medias, 1 - alfa / 2))


def bootstrap_pareado(a: Sequence[float], b: Sequence[float], n: int = 10_000, semente: int = 0) -> float:
    """Fração das reamostragens em que A não supera B (p-valor unilateral aproximado)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(semente)
    medias = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return float((medias <= 0).mean())
