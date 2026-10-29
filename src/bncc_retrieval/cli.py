"""Sugere habilidades da BNCC para uma descrição de aula.

    bncc-sugerir "Os alunos montam uma maquete do sistema solar..." --componente CI --ano 9
    bncc-sugerir "..." --componente MA --ano 6 --llm      # rerank com Claude (precisa de ANTHROPIC_API_KEY)
"""
from __future__ import annotations

import argparse

from .bncc import COMPONENTES, carregar_corpus
from .dense import CrossEncoder, Denso
from .fusion import rrf
from .lexical import BM25


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("aula", help="descrição da aula")
    ap.add_argument("--componente", choices=sorted(COMPONENTES), help="sigla da disciplina (ex.: MA)")
    ap.add_argument("--ano", type=int, choices=range(1, 10), help="ano escolar (1 a 9)")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--llm", nargs="?", const="claude-opus-5-5", default=None)
    args = ap.parse_args()

    corpus = carregar_corpus()
    docs = [h.descricao for h in corpus]
    ok = [i for i, h in enumerate(corpus)
          if (args.componente is None or h.componente == args.componente)
          and (args.ano is None or args.ano in h.anos)]

    def ordenar(scores):
        return [corpus[i].codigo for i in sorted(ok, key=lambda i: -scores[i])]

    hibrido = rrf([ordenar(BM25().ajustar(docs).pontuar(args.aula)),
                   ordenar(Denso().ajustar(docs).pontuar(args.aula))])
    desc = {h.codigo: h.descricao for h in corpus}
    candidatos = [(c, desc[c]) for c in hibrido[:20]]

    if args.llm:
        if not (args.componente and args.ano):
            ap.error("--llm precisa de --componente e --ano")
        from .llm_rerank import RerankLLM

        final = RerankLLM(modelo=args.llm).reordenar(args.aula, args.componente, args.ano, candidatos)
    else:
        final = CrossEncoder().reordenar(args.aula, candidatos)

    for pos, codigo in enumerate(final[: args.k], start=1):
        print(f"{pos}. {codigo}  {desc[codigo]}")


if __name__ == "__main__":
    main()
