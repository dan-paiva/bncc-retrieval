import math

import pytest

from bncc_retrieval.bncc import Habilidade, anos_do_codigo, componente_do_codigo, filtrar
from bncc_retrieval.extract import extrair
from bncc_retrieval.fusion import rrf
from bncc_retrieval.lexical import BM25, Jaccard, TfIdf
from bncc_retrieval.metrics import bootstrap_ic, hit_at, ndcg_at, reciprocal_rank
from bncc_retrieval.text import tokens


@pytest.mark.parametrize("codigo,anos", [
    ("EF06MA01", (6,)), ("EF35LP01", (3, 4, 5)), ("EF67LP06", (6, 7)),
    ("EF15AR01", (1, 2, 3, 4, 5)), ("EF69LP32", (6, 7, 8, 9)), ("EF89EF19", (8, 9)),
])
def test_anos_do_codigo(codigo, anos):
    assert anos_do_codigo(codigo) == anos


def test_codigo_invalido():
    with pytest.raises(ValueError):
        anos_do_codigo("EM13LP01")
    with pytest.raises(ValueError):
        componente_do_codigo("EF06XX01")


def test_filtrar_por_componente_e_ano():
    corpus = [Habilidade("EF06MA01", "MA", (6,), "a"), Habilidade("EF67LP01", "LP", (6, 7), "b"),
              Habilidade("EF07MA01", "MA", (7,), "c")]
    assert [h.codigo for h in filtrar(corpus, "MA", 6)] == ["EF06MA01"]
    assert [h.codigo for h in filtrar(corpus, "LP", 7)] == ["EF67LP01"]
    assert len(filtrar(corpus, None, None)) == 3


def test_extrair_corta_na_linha_em_branco_e_limpa():
    texto = (
        "(EF06MA01) Comparar e orde-\nnar números na reta.\n\n115\nBASE NACIONAL\n"
        "(EF06MA02) Reconhecer o sistema de numeração e suas ﬁguras.\n\n"
        "texto da introdução cita (EF06MA01) de novo\n"
    )
    h = extrair(texto)
    assert h["EF06MA01"] == "Comparar e ordenar números na reta."  # fica a ocorrência mais longa
    assert h["EF06MA02"] == "Reconhecer o sistema de numeração e suas figuras."


def test_tokens_dobra_acentos_tira_stopwords_e_trunca():
    assert tokens("As Frações da pizza") == ["fracoe", "pizza"]
    assert tokens("multiplicação", truncar=None) == ["multiplicacao"]


DOCS = ["frações equivalentes com pizza", "sistema solar e planetas", "verbos no passado em inglês"]


@pytest.mark.parametrize("cls", [Jaccard, TfIdf, BM25])
def test_lexicos_acham_o_documento_certo(cls):
    m = cls().ajustar(DOCS)
    assert m.pontuar("os planetas do sistema solar").argmax() == 1
    assert m.pontuar("xyzw").max() == 0


def test_metricas():
    rk = ["a", "b", "c", "d"]
    assert reciprocal_rank(rk, {"c"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(rk, {"z"}) == 0
    assert hit_at(rk, {"c"}, 2) == 0 and hit_at(rk, {"c"}, 3) == 1
    assert ndcg_at(rk, {"a"}, 10) == 1
    assert ndcg_at(rk, {"b"}, 10) == pytest.approx(1 / math.log2(3))


def test_bootstrap_ic_contem_a_media():
    media, lo, hi = bootstrap_ic([0, 1] * 50)
    assert media == 0.5 and lo < 0.5 < hi


def test_rrf_premia_quem_aparece_bem_nas_duas_listas():
    assert rrf([["a", "b", "c"], ["b", "c", "a"]])[0] == "b"
