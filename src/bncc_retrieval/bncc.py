"""Códigos da BNCC (Ensino Fundamental) e o corpus de habilidades.

Um código como EF67LP06 se lê assim:
    EF  -> Ensino Fundamental
    67  -> anos a que se aplica (6º e 7º)
    LP  -> componente curricular (Língua Portuguesa)
    06  -> número da habilidade
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

CODE_RE = re.compile(r"^EF(\d{2})([A-Z]{2})(\d{2})$")

COMPONENTES = {
    "LP": "Língua Portuguesa",
    "AR": "Arte",
    "EF": "Educação Física",
    "LI": "Língua Inglesa",
    "MA": "Matemática",
    "CI": "Ciências",
    "GE": "Geografia",
    "HI": "História",
    "ER": "Ensino Religioso",
}

# Blocos de anos que não são "um ano só" (EF06 = 6º ano).
_BLOCOS = {
    "12": range(1, 3),
    "15": range(1, 6),
    "35": range(3, 6),
    "67": range(6, 8),
    "69": range(6, 10),
    "89": range(8, 10),
}

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CORPUS_CSV = DATA_DIR / "bncc_ef.csv"


def anos_do_codigo(codigo: str) -> tuple[int, ...]:
    """Anos escolares cobertos por um código. EF35LP01 -> (3, 4, 5)."""
    m = CODE_RE.match(codigo)
    if not m:
        raise ValueError(f"código BNCC inválido: {codigo!r}")
    bloco = m.group(1)
    if bloco in _BLOCOS:
        return tuple(_BLOCOS[bloco])
    ano = int(bloco)
    if not 1 <= ano <= 9:
        raise ValueError(f"ano fora do Ensino Fundamental: {codigo!r}")
    return (ano,)


def componente_do_codigo(codigo: str) -> str:
    m = CODE_RE.match(codigo)
    if not m or m.group(2) not in COMPONENTES:
        raise ValueError(f"componente desconhecido: {codigo!r}")
    return m.group(2)


@dataclass(frozen=True)
class Habilidade:
    codigo: str
    componente: str
    anos: tuple[int, ...]
    descricao: str


def carregar_corpus(path: Path = CORPUS_CSV) -> list[Habilidade]:
    with open(path, encoding="utf-8") as f:
        return [
            Habilidade(
                codigo=r["codigo"],
                componente=r["componente"],
                anos=tuple(int(a) for a in r["anos"].split("|")),
                descricao=r["descricao"],
            )
            for r in csv.DictReader(f)
        ]


def filtrar(corpus: list[Habilidade], componente: str | None, ano: int | None) -> list[Habilidade]:
    """Candidatos compatíveis com o que o professor já sabe da aula: disciplina e ano."""
    return [
        h for h in corpus
        if (componente is None or h.componente == componente)
        and (ano is None or ano in h.anos)
    ]
