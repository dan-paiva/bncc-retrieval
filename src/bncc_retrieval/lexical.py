"""Recuperação léxica do zero, em numpy: Jaccard, TF-IDF + cosseno e BM25.

Todos seguem a mesma interface: `ajustar(documentos)` uma vez sobre o corpus inteiro
(o IDF vem dos 1301 documentos) e `pontuar(consulta)` devolve um score por documento.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from .text import tokens


class Jaccard:
    nome = "jaccard"

    def ajustar(self, documentos: list[str]) -> Jaccard:
        self._conjuntos = [set(tokens(d)) for d in documentos]
        return self

    def pontuar(self, consulta: str) -> np.ndarray:
        q = set(tokens(consulta))
        return np.array([len(q & d) / len(q | d) if q | d else 0.0 for d in self._conjuntos])


class _Vocabulario:
    def _indexar(self, documentos: list[str]) -> list[Counter]:
        contagens = [Counter(tokens(d)) for d in documentos]
        self.vocab = {t: i for i, t in enumerate(sorted({t for c in contagens for t in c}))}
        self.n_docs = len(documentos)
        df = np.zeros(len(self.vocab))
        for c in contagens:
            for t in c:
                df[self.vocab[t]] += 1
        self.df = df
        return contagens


class TfIdf(_Vocabulario):
    """TF sublinear (1 + log tf), IDF suavizado, vetores normalizados: score = cosseno."""

    nome = "tfidf"

    def ajustar(self, documentos: list[str]) -> TfIdf:
        contagens = self._indexar(documentos)
        self.idf = np.log((1 + self.n_docs) / (1 + self.df)) + 1
        self.matriz = np.vstack([self._vetor(c) for c in contagens])
        return self

    def _vetor(self, contagem: Counter) -> np.ndarray:
        v = np.zeros(len(self.vocab))
        for t, tf in contagem.items():
            if t in self.vocab:
                v[self.vocab[t]] = (1 + math.log(tf)) * self.idf[self.vocab[t]]
        norma = np.linalg.norm(v)
        return v / norma if norma else v

    def pontuar(self, consulta: str) -> np.ndarray:
        return self.matriz @ self._vetor(Counter(tokens(consulta)))


class BM25(_Vocabulario):
    nome = "bm25"

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b

    def ajustar(self, documentos: list[str]) -> BM25:
        contagens = self._indexar(documentos)
        self.idf = np.log(1 + (self.n_docs - self.df + 0.5) / (self.df + 0.5))
        tamanhos = np.array([sum(c.values()) for c in contagens], dtype=float)
        norm = self.k1 * (1 - self.b + self.b * tamanhos / tamanhos.mean())
        tf = np.zeros((self.n_docs, len(self.vocab)))
        for i, c in enumerate(contagens):
            for t, n in c.items():
                tf[i, self.vocab[t]] = n
        # Peso de cada termo em cada documento já saturado; a consulta só soma colunas.
        self.pesos = tf * (self.k1 + 1) / (tf + norm[:, None]) * self.idf
        return self

    def pontuar(self, consulta: str) -> np.ndarray:
        cols = [self.vocab[t] for t in set(tokens(consulta)) if t in self.vocab]
        return self.pesos[:, cols].sum(axis=1) if cols else np.zeros(self.n_docs)
