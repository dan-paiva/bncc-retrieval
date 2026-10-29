"""Benchmark: descrições de aula sintéticas, cada uma escrita a partir de uma habilidade sorteada.

Protocolo completo em docs/benchmark.md. Resumo:
- sorteio estratificado por componente (proporcional ao tamanho, mínimo de 10), semente fixa;
- cada aula foi escrita como um professor descreveria o plano, sem copiar a frase da habilidade;
- divisão fixa em dev (para ajustar) e test (só para o número final).
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .bncc import DATA_DIR, Habilidade

CONSULTAS_JSONL = DATA_DIR / "consultas.jsonl"
SEMENTE = 42


@dataclass(frozen=True)
class Consulta:
    id: str
    componente: str
    ano: int
    texto: str
    gold: tuple[str, ...]
    split: str


def sortear(corpus: list[Habilidade], n: int = 300, minimo: int = 10, semente: int = SEMENTE) -> list[dict]:
    """Habilidades sorteadas para virar aula, com o ano concreto da aula e o split."""
    rng = random.Random(semente)
    por_comp: dict[str, list[Habilidade]] = defaultdict(list)
    for h in corpus:
        por_comp[h.componente].append(h)
    cotas = {c: max(minimo, round(n * len(hs) / len(corpus))) for c, hs in por_comp.items()}
    # Ajusta o total para n tirando dos maiores componentes.
    while sum(cotas.values()) > n:
        cotas[max(cotas, key=cotas.get)] -= 1
    escolhidas = []
    for comp in sorted(por_comp):
        for h in rng.sample(por_comp[comp], cotas[comp]):
            escolhidas.append({"codigo": h.codigo, "componente": comp, "ano": rng.choice(h.anos),
                               "descricao": h.descricao})
    rng.shuffle(escolhidas)
    for i, e in enumerate(escolhidas):
        e["id"] = f"q{i + 1:03d}"
        e["split"] = "dev" if i < n // 3 else "test"
    return escolhidas


def carregar(path: Path = CONSULTAS_JSONL, split: str | None = None) -> list[Consulta]:
    consultas = []
    with open(path, encoding="utf-8") as f:
        for linha in f:
            r = json.loads(linha)
            if split and r["split"] != split:
                continue
            consultas.append(Consulta(r["id"], r["componente"], r["ano"], r["texto"], tuple(r["gold"]), r["split"]))
    return consultas
