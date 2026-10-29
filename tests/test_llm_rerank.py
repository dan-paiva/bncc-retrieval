"""O rerank com LLM testado sem rede: um cliente falso imita a resposta da API."""
import json
from types import SimpleNamespace

from bncc_retrieval.llm_rerank import RerankLLM

CANDIDATOS = [("EF06MA01", "comparar números"), ("EF06MA02", "sistema de numeração"),
              ("EF06MA03", "operações")]


class ClienteFalso:
    def __init__(self, ranking, stop_reason="end_turn"):
        self.pedidos = []
        self.ranking, self.stop_reason = ranking, stop_reason
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.pedidos.append(kw)
        texto = json.dumps({"ranking": self.ranking, "justificativa": "x"})
        return SimpleNamespace(
            stop_reason=self.stop_reason, model=kw["model"],
            content=[SimpleNamespace(type="text", text=texto)],
            usage=SimpleNamespace(input_tokens=1000, output_tokens=50),
        )


def _rerank(tmp_path, cliente):
    return RerankLLM(modelo="claude-opus-5-5", cliente=cliente, cache=tmp_path / "cache.jsonl")


def test_schema_so_aceita_os_candidatos(tmp_path):
    cliente = ClienteFalso(["EF06MA02"])
    _rerank(tmp_path, cliente).reordenar("aula", "MA", 6, CANDIDATOS)
    schema = cliente.pedidos[0]["output_config"]["format"]["schema"]
    assert schema["properties"]["ranking"]["items"]["enum"] == ["EF06MA01", "EF06MA02", "EF06MA03"]


def test_descarta_codigo_inventado_e_completa_o_ranking(tmp_path):
    cliente = ClienteFalso(["EF06MA03", "EF99XX99", "EF06MA03"])
    rk = _rerank(tmp_path, cliente).reordenar("aula", "MA", 6, CANDIDATOS)
    assert rk == ["EF06MA03", "EF06MA01", "EF06MA02"]


def test_cache_evita_segunda_chamada_e_conta_custo(tmp_path):
    cliente = ClienteFalso(["EF06MA02"])
    r = _rerank(tmp_path, cliente)
    r.reordenar("aula", "MA", 6, CANDIDATOS)
    r2 = _rerank(tmp_path, cliente)  # nova instância lê o cache do disco
    r2.reordenar("aula", "MA", 6, CANDIDATOS)
    assert len(cliente.pedidos) == 1
    assert r2.gasto.do_cache == 1
    assert r.gasto.dolares == (1000 * 4.00 + 50 * 20.00) / 1e6


def test_recusa_mantem_a_ordem_da_primeira_etapa(tmp_path):
    rk = _rerank(tmp_path, ClienteFalso([], stop_reason="refusal")).reordenar("aula", "MA", 6, CANDIDATOS)
    assert rk == ["EF06MA01", "EF06MA02", "EF06MA03"]
