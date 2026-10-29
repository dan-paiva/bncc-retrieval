"""Recuperação densa (bi-encoder) e rerank com cross-encoder, rodando local (GPU se houver).

Modelos multilíngues, sem fine-tuning:
- bi-encoder:    intfloat/multilingual-e5-base  (exige os prefixos "query: " e "passage: ")
- cross-encoder: BAAI/bge-reranker-v2-m3
Os embeddings do corpus ficam em cache em .cache/ (o corpus só muda se o CSV mudar).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parents[2] / ".cache"


def _kwargs() -> dict:
    """fp16 na GPU: os dois modelos juntos cabem em 4 GB (fp32 não cabe)."""
    import torch

    return {"torch_dtype": torch.float16} if torch.cuda.is_available() else {}


class Denso:
    nome = "e5"

    def __init__(self, modelo: str = "intfloat/multilingual-e5-base"):
        from sentence_transformers import SentenceTransformer

        self.modelo_id = modelo
        self.modelo = SentenceTransformer(modelo, model_kwargs=_kwargs())

    def ajustar(self, documentos: list[str]) -> Denso:
        chave = hashlib.sha256((self.modelo_id + "\n".join(documentos)).encode()).hexdigest()[:16]
        arquivo = CACHE / f"emb_{chave}.npy"
        if arquivo.exists():
            self.matriz = np.load(arquivo)
        else:
            self.matriz = self.modelo.encode([f"passage: {d}" for d in documentos], batch_size=32,
                                             normalize_embeddings=True, show_progress_bar=False)
            CACHE.mkdir(exist_ok=True)
            np.save(arquivo, self.matriz)
        return self

    def pontuar(self, consulta: str) -> np.ndarray:
        q = self.modelo.encode([f"query: {consulta}"], normalize_embeddings=True)[0]
        return self.matriz @ q


class CrossEncoder:
    nome = "cross-encoder"

    def __init__(self, modelo: str = "BAAI/bge-reranker-v2-m3"):
        from sentence_transformers import CrossEncoder as _CE

        self.modelo = _CE(modelo, max_length=512, model_kwargs=_kwargs())

    def reordenar(self, consulta: str, candidatos: list[tuple[str, str]]) -> list[str]:
        """candidatos: [(código, descrição)] -> códigos do mais ao menos relevante."""
        scores = self.modelo.predict([(consulta, d) for _, d in candidatos], batch_size=8,
                                     show_progress_bar=False)
        ordem = np.argsort(-np.asarray(scores), kind="stable")
        return [candidatos[i][0] for i in ordem]
