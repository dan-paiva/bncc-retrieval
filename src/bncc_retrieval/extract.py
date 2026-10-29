"""Extrai as habilidades do Ensino Fundamental do PDF oficial da BNCC.

Fonte: BNCC homologada (MEC, dez/2017), http://basenacionalcomum.mec.gov.br/images/BNCC_20dez_site.pdf
Uso:   python -m bncc_retrieval.extract caminho/BNCC_20dez_site.pdf

Estratégia: `pdftotext` (poppler) em modo de fluxo. Cada habilidade começa com o código
entre parênteses e termina na primeira linha em branco (o PDF separa blocos assim).
Quando um código aparece mais de uma vez, fica a ocorrência com a descrição mais longa.
"""
from __future__ import annotations

import csv
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

from .bncc import CORPUS_CSV, anos_do_codigo, componente_do_codigo

MARCADOR = re.compile(r"\((EF\d{2}[A-Z]{2}\d{2})\)")
# Ligaduras tipográficas que o PDF usa (ﬁ, ﬂ) e hifenização de fim de linha.
LIGADURAS = str.maketrans({"ﬁ": "fi", "ﬂ": "fl"})
HIFEN_QUEBRA = re.compile(r"(\w)-\n(\w)")


def limpar(trecho: str) -> str:
    trecho = trecho.split("\n\n", 1)[0]
    trecho = HIFEN_QUEBRA.sub(r"\1\2", trecho.translate(LIGADURAS))
    trecho = unicodedata.normalize("NFC", trecho)
    return re.sub(r"\s+", " ", trecho).strip()


def extrair(texto: str) -> dict[str, str]:
    """Texto do PDF -> {código: descrição}."""
    marcas = list(MARCADOR.finditer(texto))
    habilidades: dict[str, str] = {}
    for atual, proxima in zip(marcas, marcas[1:] + [None], strict=True):
        fim = proxima.start() if proxima else len(texto)
        descricao = limpar(texto[atual.end():fim])
        codigo = atual.group(1)
        if len(descricao) > len(habilidades.get(codigo, "")):
            habilidades[codigo] = descricao
    return habilidades


def pdf_para_texto(pdf: Path) -> str:
    return subprocess.run(
        ["pdftotext", str(pdf), "-"], capture_output=True, text=True, check=True
    ).stdout


def salvar(habilidades: dict[str, str], destino: Path = CORPUS_CSV) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    with open(destino, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["codigo", "componente", "anos", "descricao"])
        for codigo in sorted(habilidades):
            anos = "|".join(map(str, anos_do_codigo(codigo)))
            w.writerow([codigo, componente_do_codigo(codigo), anos, habilidades[codigo]])


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("uso: python -m bncc_retrieval.extract BNCC_20dez_site.pdf")
    habilidades = extrair(pdf_para_texto(Path(sys.argv[1])))
    salvar(habilidades)
    print(f"{len(habilidades)} habilidades -> {CORPUS_CSV}")


if __name__ == "__main__":
    main()
