"""Normalização e tokenização para português.

Stemming por truncamento: cortar cada palavra nos 6 primeiros caracteres junta flexões
("multiplicar", "multiplicação" -> "multip"; "geométrica", "geometria" -> "geomet").
É grosseiro e às vezes falha ("fração" -> "fracao", "frações" -> "fracoe"), mas é conhecido
por funcionar bem em recuperação em português e não tem dependência.
"""
from __future__ import annotations

import re
import unicodedata

STOPWORDS = frozenset("""
a ao aos as ate cada com como da das de dela dele deles depois do dos e ela elas ele eles em
entao entre era essa esse esta estao estar este eu foi for ha isso ja la lhe mais mas me mesmo
muito na nas nao nem no nos nossa nosso num numa o os ou para pela pelas pelo pelos por qual
quando que quem se sem ser seu sua suas seus so sobre tambem tem ter todo toda todos todas tu
um uma umas uns vai vez voce cujo cuja etc bem ainda onde assim apos antes outro outra outros
outras aqui ali
""".split())

_PALAVRA = re.compile(r"[a-z0-9]+")


def dobrar(texto: str) -> str:
    """Minúsculas e sem acentos: 'Frações' -> 'fracoes'."""
    sem_acento = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in sem_acento if not unicodedata.combining(c))


def tokens(texto: str, truncar: int | None = 6) -> list[str]:
    saida = []
    for p in _PALAVRA.findall(dobrar(texto)):
        if len(p) < 3 or p in STOPWORDS:
            continue
        saida.append(p[:truncar] if truncar else p)
    return saida
