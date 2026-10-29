"""Roda o benchmark e escreve results/RESULTADOS.md.

    bncc-avaliar --split test                 # todos os métodos locais
    bncc-avaliar --split test --llm           # + rerank com Claude (usa cache; chama a API só no que faltar)
    bncc-avaliar --split dev --metodos bm25 e5

Cenários:
    filtrado  o professor já informou disciplina e ano: só concorrem as habilidades compatíveis (~10-60)
    aberto    nada informado: concorrem as 1301 habilidades
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .benchmark import Consulta, carregar
from .bncc import Habilidade, carregar_corpus
from .fusion import rrf
from .lexical import BM25, Jaccard, TfIdf
from .metrics import bootstrap_ic, bootstrap_pareado, por_consulta

RESULTS = Path(__file__).resolve().parents[2] / "results"
PRIMEIRA_ETAPA = 20  # quantos candidatos o rerank (cross-encoder ou LLM) recebe
METRICAS = ["mrr", "hit@1", "hit@3", "hit@5", "hit@10", "ndcg@10"]


class Avaliador:
    def __init__(self, corpus: list[Habilidade], metodos: list[str], llm_modelo: str | None):
        self.corpus = corpus
        self.codigos = np.array([h.codigo for h in corpus])
        docs = [h.descricao for h in corpus]
        self.base = {}
        for cls in (Jaccard, TfIdf, BM25):
            if cls.nome in metodos or "hibrido" in " ".join(metodos):
                self.base[cls.nome] = cls().ajustar(docs)
        self.denso = self.ce = self.llm = None
        if any(m.startswith(("e5", "hibrido")) for m in metodos):
            from .dense import Denso

            self.denso = Denso().ajustar(docs)
        if "hibrido+cross-encoder" in metodos:
            from .dense import CrossEncoder

            self.ce = CrossEncoder()
        if llm_modelo:
            from .llm_rerank import RerankLLM

            self.llm = RerankLLM(modelo=llm_modelo)
        self.descricao = {h.codigo: h.descricao for h in corpus}

    def _mascara(self, c: Consulta, cenario: str) -> np.ndarray:
        if cenario == "aberto":
            return np.ones(len(self.corpus), dtype=bool)
        return np.array([h.componente == c.componente and c.ano in h.anos for h in self.corpus])

    def _ordenar(self, scores: np.ndarray, mascara: np.ndarray) -> list[str]:
        idx = np.flatnonzero(mascara)
        return list(self.codigos[idx[np.argsort(-scores[idx], kind="stable")]])

    def rankings(self, c: Consulta, cenario: str, metodos: list[str]) -> tuple[dict, dict]:
        """Ranking e latência (ms) de cada método para uma consulta."""
        mascara = self._mascara(c, cenario)
        saida, ms = {}, {}

        def medir(nome, fn):
            t = time.perf_counter()
            saida[nome] = fn()
            ms[nome] = (time.perf_counter() - t) * 1000

        for nome, m in self.base.items():
            medir(nome, lambda m=m: self._ordenar(m.pontuar(c.texto), mascara))
        if self.denso:
            medir("e5", lambda: self._ordenar(self.denso.pontuar(c.texto), mascara))
            medir("hibrido", lambda: rrf([saida["bm25"], saida["e5"]]))
            ms["hibrido"] += ms["bm25"] + ms["e5"]
        candidatos = [(cod, self.descricao[cod]) for cod in saida.get("hibrido", [])[:PRIMEIRA_ETAPA]]
        cauda = saida.get("hibrido", [])[PRIMEIRA_ETAPA:]
        if self.ce:
            medir("hibrido+cross-encoder", lambda: self.ce.reordenar(c.texto, candidatos) + cauda)
            ms["hibrido+cross-encoder"] += ms["hibrido"]
        if self.llm:
            medir(self.llm.nome, lambda: self.llm.reordenar(c.texto, c.componente, c.ano, candidatos) + cauda)
            ms[self.llm.nome] += ms["hibrido"]
        return {k: v for k, v in saida.items() if k in metodos or k == getattr(self.llm, "nome", None)}, ms


def _tabela(resumo: dict, ordem: list[str]) -> list[str]:
    linhas = ["| Método | MRR (IC 95%) | Hit@1 | Hit@3 | Hit@5 | Hit@10 | nDCG@10 | ms/consulta |",
              "|---|---|---|---|---|---|---|---|"]
    for nome in ordem:
        r = resumo[nome]
        mrr, lo, hi = r["mrr"]
        celulas = [f"**{mrr:.3f}** ({lo:.3f}–{hi:.3f})"] + [f"{r[m][0]:.1%}" for m in METRICAS[1:5]]
        celulas += [f"{r['ndcg@10'][0]:.3f}", f"{r['ms']:.0f}"]
        linhas.append(f"| {nome} | " + " | ".join(celulas) + " |")
    return linhas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["dev", "test", "todos"])
    ap.add_argument("--metodos", nargs="+",
                    default=["jaccard", "tfidf", "bm25", "e5", "hibrido", "hibrido+cross-encoder"])
    ap.add_argument("--llm", nargs="?", const="claude-opus-5-5", default=None,
                    help="rerank com Claude (modelo opcional, padrão claude-opus-5-5)")
    ap.add_argument("--cenarios", nargs="+", default=["filtrado", "aberto"])
    args = ap.parse_args()

    consultas = carregar(split=None if args.split == "todos" else args.split)
    av = Avaliador(carregar_corpus(), args.metodos, args.llm)
    relatorio = {"split": args.split, "n": len(consultas), "cenarios": {}}
    md = [f"# Resultados ({args.split}, {len(consultas)} consultas)", ""]

    for cenario in args.cenarios:
        por_metodo = defaultdict(lambda: defaultdict(list))
        latencias = defaultdict(list)
        por_comp = defaultdict(lambda: defaultdict(list))
        for c in consultas:
            rks, ms = av.rankings(c, cenario, args.metodos)
            for nome, rk in rks.items():
                for m, v in por_consulta(rk, c.gold).items():
                    por_metodo[nome][m].append(v)
                por_comp[nome][c.componente].append(por_consulta(rk, c.gold)["hit@3"])
                latencias[nome].append(ms[nome])

        ordem = list(por_metodo)
        resumo = {n: {m: bootstrap_ic(por_metodo[n][m]) for m in METRICAS} | {"ms": float(np.median(latencias[n]))}
                  for n in ordem}
        melhor = max(ordem, key=lambda n: resumo[n]["mrr"][0])
        relatorio["cenarios"][cenario] = {"resumo": resumo, "por_consulta": {n: dict(por_metodo[n]) for n in ordem}}

        md += [f"## Cenário: {cenario}", ""] + _tabela(resumo, ordem) + [""]
        if "bm25" in ordem and melhor != "bm25":
            p = bootstrap_pareado(por_metodo[melhor]["mrr"], por_metodo["bm25"]["mrr"])
            md.append(f"Melhor método: **{melhor}**. Bootstrap pareado contra BM25 (MRR): p ≈ {p:.4f}.")
        md += ["", f"Hit@3 por componente ({melhor}):", "",
               "| Componente | n | Hit@3 |", "|---|---|---|"]
        for comp, vals in sorted(por_comp[melhor].items()):
            md.append(f"| {comp} | {len(vals)} | {np.mean(vals):.0%} |")
        md.append("")

    if av.llm:
        g = av.llm.gasto
        md += ["## Custo do rerank com LLM", "",
               f"- Modelo: `{av.llm.modelo}`, esforço `{av.llm.esforco}`, {PRIMEIRA_ETAPA} candidatos por consulta",
               f"- Respostas: {g.chamadas + g.do_cache} ({g.chamadas} chamadas novas, {g.do_cache} do cache)",
               f"- Tokens: {g.tokens_entrada:,} de entrada, {g.tokens_saida:,} de saída",
               f"- Custo total equivalente: US$ {g.dolares:.2f} "
               f"(US$ {g.dolares / max(1, g.chamadas + g.do_cache) * 1000:.2f} por mil consultas)", ""]
        relatorio["custo_llm"] = {"tokens_entrada": g.tokens_entrada, "tokens_saida": g.tokens_saida,
                                  "dolares": g.dolares}

    RESULTS.mkdir(exist_ok=True)
    sufixo = args.split
    (RESULTS / f"metricas_{sufixo}.json").write_text(json.dumps(relatorio, ensure_ascii=False, indent=1))
    (RESULTS / f"RESULTADOS_{sufixo}.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
